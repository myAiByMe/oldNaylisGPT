import os
os.environ["TORCHINDUCTOR_CACHE_DIR"]      = "./CompileCache"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.makedirs("./CompileCache", exist_ok=True)

import sys
import time
import math
import json
import gc
import traceback
import argparse
import threading
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from functools import partial
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Optional, List

torch.set_float32_matmul_precision('high')

# ── Paths Core ───────────────────────────────────────────────────────────────
_root = os.path.dirname(__file__)
sys.path.append(os.path.join(_root, 'Core', 'Model'))
sys.path.append(os.path.join(_root, 'Core', 'Attention'))
sys.path.append(os.path.join(_root, 'Core', 'FeedForward'))
sys.path.append(os.path.join(_root, 'Core', 'TransformerBlock'))

from HessGpt import NaylisGPT
from attention import KVCache

# ── Args ─────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--no-compile',   action='store_true')
    p.add_argument('--compile-mode', default='default',
                   choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--total-steps',  type=int, default=None,
                   help='Forcer total_steps du scheduler WSD (reprise mid-training).')
    p.add_argument('--HF_token',     type=str, default=None,
                   help='Token Hugging Face (lecture/écriture). Ex: hf_xxxxxxxx')
    return p.parse_args()

ARGS   = get_args()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Hugging Face ──────────────────────────────────────────────────────────────
HF_DATASET_REPO          = "silyan/data_PoC"       # lecture data
HF_MODEL_REPO            = "silyan/testOldScript"  # écriture modèle + cache
HF_TOKEN                 = ARGS.HF_token
HF_TIMED_SAVE_EVERY_MIN  = 60

COMPILE_CACHE_DIR  = "./CompileCache"
COMPILE_CACHE_ZIP  = "CompileCache.zip"

if HF_TOKEN is None:
    print('  ⚠️  Aucun token HF fourni (--HF_token). '
          'Les repos privés et les pushs seront désactivés.')

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    # Modèle
    'vocab_size'            : None,
    'embed_dim'             : 512,
    'num_heads'             : 8,
    'num_layers'            : 12,
    'max_seq_len'           : 512,
    'dropout'               : 0.0,
    'use_rope'              : True,
    'use_yarn'              : False,
    'yarn_scale'            : 1.0,
    'yarn_original_max_len' : 512,
    'use_swiglu'            : True,
    'n_kv_heads'            : 4,
    'use_qk_norm'           : True,
    'soft_cap'              : None,
    'use_flash_attn'        : True,
    'rel_rank'              : 8,
    # Training
    'batch_size'            : 190,
    'gradient_accumulation' : 1,
    'max_grad_norm'         : 1.0,
    'learning_rate'         : 3e-4,
    'weight_decay'          : 0.1,
    'adam_beta1'            : 0.9,
    'adam_beta2'            : 0.95,
    'adam_eps'              : 1e-8,
    # Data
    'data_dir'              : './data_exp',
    'val_tokens'            : 10_000_000,
    'warmup_ratio'          : 0.03,
    'decay_ratio'           : 0.15,
    'min_lr_ratio'          : 0.1,
    # Validation / Save
    'validate_every_steps'  : 500,
    'val_batches'           : 50,
    'save_every_steps'      : 2000,
    # Checkpoint
    'checkpoint_file'       : './Model/naylis_pretrain.pt',
    # Compile
    'use_compile'           : not ARGS.no_compile,
    'compile_mode'          : ARGS.compile_mode,
    # DataLoader
    'num_workers'           : 1,
    'use_packing'           : True,
}

print('=' * 70)
print('  Naylis v1 — Pretrain')
print('=' * 70)
if DEVICE == 'cuda':
    print(f'  GPU  : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
    cap = torch.cuda.get_device_capability()
    print(f'  SM   : {cap[0]}{cap[1]}')
print(f'  embed={CONFIG["embed_dim"]}  layers={CONFIG["num_layers"]}  '
      f'heads={CONFIG["num_heads"]}  kv={CONFIG["n_kv_heads"]}  rel_rank={CONFIG["rel_rank"]}  '
      f'batch_size={CONFIG["batch_size"]}  grad_acc={CONFIG["gradient_accumulation"]}')


# ── Tokenizer ────────────────────────────────────────────────────────────────
print('\nTokenizer...')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/cosmo2-tokenizer')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
CONFIG['vocab_size'] = len(tokenizer)
print(f'  vocab={len(tokenizer)}  eos={tokenizer.eos_token_id}')


# ── HF helpers ───────────────────────────────────────────────────────────────

def hf_list_files(repo_id: str, repo_type: str = "dataset") -> list:
    """Liste les fichiers d'un repo HF. Retourne [] si erreur."""
    try:
        from huggingface_hub import list_repo_files
        return list(list_repo_files(repo_id=repo_id, repo_type=repo_type, token=HF_TOKEN))
    except Exception as e:
        print(f'  ⚠️  Impossible de lister {repo_id} : {e}')
        return []


def hf_check_and_download_data(data_dir: str):
    """Si data_dir ne contient pas de chunks, télécharge depuis HF_DATASET_REPO."""
    from huggingface_hub import snapshot_download
    existing = [e for e in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, e)) and e.startswith('chunk')]
    if existing:
        print(f'  ✅ Data déjà présente : {len(existing)} chunk(s) dans {data_dir}')
        return
    print(f'\n{"="*70}')
    print(f'  📥 DATA ABSENTE — téléchargement depuis {HF_DATASET_REPO}')
    print(f'{"="*70}')
    try:
        snapshot_download(
            repo_id        = HF_DATASET_REPO,
            repo_type      = "dataset",
            local_dir      = data_dir,
            token          = HF_TOKEN,
            ignore_patterns= ["*.md", "*.gitattributes"],
        )
        print(f'  ✅ Dataset téléchargé dans {data_dir}')
    except Exception as e:
        print(f'  ❌ Échec téléchargement dataset : {e}')
        sys.exit(1)


def hf_check_and_download_checkpoint(ckpt_path: str):
    """Télécharge le checkpoint depuis HF_MODEL_REPO si absent localement."""
    from huggingface_hub import hf_hub_download
    ckpt_dir      = os.path.dirname(ckpt_path) or '.'
    ckpt_filename = os.path.basename(ckpt_path)
    info_filename = ckpt_filename.replace('.pt', '_info.json')
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.path.exists(ckpt_path):
        print(f'  ✅ Checkpoint local trouvé : {ckpt_path}')
        return

    print(f'  🔍 Checkpoint absent — recherche sur {HF_MODEL_REPO}...')
    remote_files = hf_list_files(HF_MODEL_REPO)
    for fname in (ckpt_filename, info_filename):
        if fname not in remote_files:
            print(f'  ℹ️  {fname} absent sur HF → nouveau training')
            continue
        try:
            hf_hub_download(
                repo_id   = HF_MODEL_REPO,
                filename  = fname,
                repo_type = "dataset",
                token     = HF_TOKEN,
                local_dir = ckpt_dir,
            )
            print(f'  ✅ {fname} téléchargé depuis HF')
        except Exception as e:
            print(f'  ⚠️  Échec download {fname} : {e}')


def hf_download_compile_cache():
    """
    Télécharge CompileCache.zip depuis HF_MODEL_REPO et le dézippe localement.
    Appelé une seule fois au démarrage.
    """
    from huggingface_hub import hf_hub_download
    remote_files = hf_list_files(HF_MODEL_REPO)
    if COMPILE_CACHE_ZIP not in remote_files:
        print(f'  ℹ️  {COMPILE_CACHE_ZIP} absent sur HF → compilation from scratch')
        return
    print(f'  📥 Téléchargement {COMPILE_CACHE_ZIP} depuis {HF_MODEL_REPO}...')
    try:
        local_zip = hf_hub_download(
            repo_id   = HF_MODEL_REPO,
            filename  = COMPILE_CACHE_ZIP,
            repo_type = "dataset",
            token     = HF_TOKEN,
            local_dir = ".",
        )
        with zipfile.ZipFile(local_zip, 'r') as zf:
            zf.extractall(".")
        print(f'  ✅ {COMPILE_CACHE_ZIP} dézippé → {COMPILE_CACHE_DIR}/')
    except Exception as e:
        print(f'  ⚠️  Échec download/unzip CompileCache : {e}')


def _zip_compile_cache() -> Optional[str]:
    """Zippe ./CompileCache/ → CompileCache.zip. Retourne le chemin ou None."""
    if not os.path.isdir(COMPILE_CACHE_DIR):
        return None
    try:
        with zipfile.ZipFile(COMPILE_CACHE_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(COMPILE_CACHE_DIR):
                for file in files:
                    fp  = os.path.join(root, file)
                    arc = os.path.relpath(fp, start=".")
                    zf.write(fp, arc)
        return COMPILE_CACHE_ZIP
    except Exception as e:
        print(f'  ⚠️  Erreur zip CompileCache : {e}')
        return None


def hf_push_checkpoint(ckpt_path: str, global_step: int, push_cache: bool = True):
    """
    Pousse le checkpoint + _info.json + CompileCache.zip vers HF_MODEL_REPO.
    Appelé en thread daemon pour ne pas bloquer le training.
    """
    if HF_TOKEN is None:
        return
    from huggingface_hub import upload_file

    ckpt_filename = os.path.basename(ckpt_path)
    info_filename = ckpt_filename.replace('.pt', '_info.json')
    info_path     = ckpt_path.replace('.pt', '_info.json')

    files_to_push = [(ckpt_path, ckpt_filename)]
    if os.path.exists(info_path):
        files_to_push.append((info_path, info_filename))

    # Zip du CompileCache
    if push_cache:
        zip_path = _zip_compile_cache()
        if zip_path:
            files_to_push.append((zip_path, COMPILE_CACHE_ZIP))

    print(f'\n  ☁️  Push HF (step={global_step:,}) → {HF_MODEL_REPO}...')
    for local_f, remote_f in files_to_push:
        if not os.path.exists(local_f):
            continue
        try:
            upload_file(
                path_or_fileobj = local_f,
                path_in_repo    = remote_f,
                repo_id         = HF_MODEL_REPO,
                repo_type       = "dataset",
                token           = HF_TOKEN,
                commit_message  = f"checkpoint step={global_step:,}",
            )
            print(f'  ✅ {remote_f} pushé')
        except Exception as e:
            print(f'  ⚠️  Push échoué pour {remote_f} : {e}')


# ── Timed save state ─────────────────────────────────────────────────────────
_timed_save_state = {
    'last_save_time': time.time(),
}

def _should_timed_save() -> bool:
    return (time.time() - _timed_save_state['last_save_time']) >= HF_TIMED_SAVE_EVERY_MIN * 60

def _mark_timed_save_done():
    _timed_save_state['last_save_time'] = time.time()

def timed_hf_save(ckpt_mgr, model, optimizers, scheduler, metadata: dict):
    """Sauvegarde locale + push HF si le timer est écoulé (thread daemon)."""
    if not _should_timed_save():
        return
    print(f'\n  ⏱️  {HF_TIMED_SAVE_EVERY_MIN} min écoulées — sauvegarde + push HF...')
    ckpt_mgr.save(model, optimizers, scheduler, metadata)
    _mark_timed_save_done()
    step = metadata.get('global_step', 0)
    t = threading.Thread(
        target=hf_push_checkpoint,
        args=(ckpt_mgr.path, step, True),
        daemon=True,
    )
    t.start()


# ── Startup : data + checkpoint + compile cache ───────────────────────────────
print('\n' + '=' * 70)
print('  🔎 VÉRIFICATION DATA, CHECKPOINT & COMPILE CACHE')
print('=' * 70)
os.makedirs(CONFIG['data_dir'], exist_ok=True)
hf_check_and_download_data(CONFIG['data_dir'])
hf_check_and_download_checkpoint(CONFIG['checkpoint_file'])
hf_download_compile_cache()


# ── Chunk scan ───────────────────────────────────────────────────────────────
def scan_chunks(data_dir: str) -> list:
    available = []
    if not os.path.exists(data_dir):
        return available
    for entry in sorted(os.listdir(data_dir)):
        chunk_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(chunk_dir) or not entry.startswith('chunk'):
            continue
        npy_file = os.path.join(chunk_dir, 'tokens.npy')
        if not os.path.exists(npy_file):
            npy_file = os.path.join(chunk_dir, 'cosmopedia.npy')
        if not os.path.exists(npy_file):
            continue
        try:
            arr = np.load(npy_file, mmap_mode='r')
            cid = int(entry.split('_')[1])
            available.append({'id': cid, 'dir': chunk_dir, 'file': npy_file,
                               'tokens': len(arr)})
        except Exception as e:
            print(f'  skip {entry}: {e}')
    available.sort(key=lambda x: x['id'])
    return available


ALL_CHUNKS = scan_chunks(CONFIG['data_dir'])
if not ALL_CHUNKS:
    print(f'\nERREUR : aucun chunk dans {CONFIG["data_dir"]}')
    sys.exit(1)

print(f'\n  {len(ALL_CHUNKS)} chunks disponibles :')
for c in ALL_CHUNKS:
    print(f'    chunk_{c["id"]:03d} : {c["tokens"] / 1e9:.3f}B tokens')


def steps_for_chunk(n_tokens: int) -> int:
    samples = n_tokens // (CONFIG['max_seq_len'] + 1)
    batches = math.ceil(samples / CONFIG['batch_size'])
    return max(math.ceil(batches / CONFIG['gradient_accumulation']), 1)


TOTAL_STEPS = sum(steps_for_chunk(c['tokens']) for c in ALL_CHUNKS)
print(f'  Total steps estimés (tous chunks) : {TOTAL_STEPS:,}')


# ── WSD Scheduler ────────────────────────────────────────────────────────────
class WSDScheduler:
    def __init__(self, optimizers, max_lr, total_steps,
                 warmup_ratio=0.03, decay_ratio=0.15, min_lr_ratio=0.1):
        self.optimizers   = optimizers if isinstance(optimizers, list) else [optimizers]
        self.max_lr       = max_lr
        self.min_lr       = max_lr * min_lr_ratio
        self.total_steps  = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps  = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.current_step = 0

    def get_lr(self) -> float:
        s = self.current_step
        if s < self.warmup_steps:
            return self.max_lr * (s / max(self.warmup_steps, 1))
        elif s < self.warmup_steps + self.stable_steps:
            return self.max_lr
        else:
            d = s - self.warmup_steps - self.stable_steps
            p = min(d / max(self.decay_steps, 1), 1.0)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * p))

    def step(self) -> float:
        lr = self.get_lr()
        self.current_step += 1
        for opt in self.optimizers:
            for pg in opt.param_groups:
                pg['lr'] = lr * 5.0 if pg.get('is_muon', False) else lr
        return lr

    def get_last_lr(self): return [self.get_lr()]
    def state_dict(self):  return {'current_step': self.current_step}
    def load_state_dict(self, sd): self.current_step = sd['current_step']


# ── Datasets ─────────────────────────────────────────────────────────────────
class ChunkDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        n = len(tokens) // (seq_len + 1)
        self.tokens  = tokens[:n * (seq_len + 1)].share_memory_()
        self.seq_len = seq_len
        self.n       = n

    def __len__(self): return self.n

    def __getitem__(self, idx):
        s = idx * (self.seq_len + 1)
        c = self.tokens[s:s + self.seq_len + 1].long()
        return c[:-1].clone(), c[1:].clone()


class PackedChunkDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int, eos_token_id: int):
        n = len(tokens) // (seq_len + 1)
        self.tokens       = tokens[:n * (seq_len + 1)].share_memory_()
        self.seq_len      = seq_len
        self.eos_token_id = eos_token_id
        self.n            = n

    def __len__(self): return self.n

    def __getitem__(self, idx):
        s = idx * (self.seq_len + 1)
        b = self.tokens[s:s + self.seq_len + 1].long()
        return b[:-1].clone(), b[1:].clone()


def packed_collate_fn(batch, eos_token_id: int, seq_len: int):
    xs, ys = zip(*batch)
    x = torch.stack(xs)
    y = torch.stack(ys)
    all_cu = [0]; max_sl = 1
    for i in range(x.size(0)):
        seq     = x[i]
        eos_pos = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_pos) == 0:
            all_cu.append(all_cu[-1] + seq_len); max_sl = max(max_sl, seq_len)
        else:
            prev = 0
            for pos in eos_pos.tolist():
                l = pos - prev + 1
                if l > 0: all_cu.append(all_cu[-1] + l); max_sl = max(max_sl, l)
                prev = pos + 1
            rem = seq_len - prev
            if rem > 0: all_cu.append(all_cu[-1] + rem); max_sl = max(max_sl, rem)
    return x, y, torch.tensor(all_cu, dtype=torch.int32), max_sl


class LazyChunk:
    def __init__(self, chunk_info: dict, seq_len: int, val_tokens: int):
        print(f'  Chargement chunk_{chunk_info["id"]:03d}...')
        t0 = time.time()
        try:
            arr  = np.load(chunk_info['file'])
            mode = 'RAM'
        except MemoryError:
            print('  MemoryError → fallback mmap')
            arr  = np.load(chunk_info['file'], mmap_mode='r')
            mode = 'mmap'

        tokens    = torch.from_numpy(arr.astype(np.int32))
        del arr; gc.collect()
        total     = len(tokens)
        seq_len_1 = seq_len + 1
        n_seqs    = total // seq_len_1
        ram_gb    = tokens.element_size() * total / 1e9

        rng    = np.random.default_rng(42 + chunk_info['id'])
        idx    = rng.permutation(n_seqs)
        tokens = tokens[:n_seqs * seq_len_1].reshape(n_seqs, seq_len_1)[idx].reshape(-1)

        val_size   = min(val_tokens, int(total * 0.05))
        train_size = total - val_size
        self._train = tokens[:train_size]
        self._val   = tokens[train_size:]
        print(f'  chunk_{chunk_info["id"]:03d} [{mode}] : {total / 1e6:.0f}M tokens  '
              f'train={train_size / 1e6:.0f}M  val={val_size / 1e6:.0f}M  '
              f'RAM={ram_gb:.2f}GB  ({time.time() - t0:.1f}s)')

    def train_dataset(self, seq_len: int, use_packing: bool, eos_id: int):
        if use_packing:
            return PackedChunkDataset(self._train, seq_len, eos_id)
        return ChunkDataset(self._train, seq_len)

    def val_dataset(self, seq_len: int):
        return ChunkDataset(self._val, seq_len)

    def unload(self):
        del self._train, self._val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('  RAM chunk libérée')


# ── Checkpoint ───────────────────────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def save(self, model, optimizers, scheduler, metadata: dict):
        m           = model._orig_mod if hasattr(model, '_orig_mod') else model
        muon, adamw = optimizers
        cp = {
            'model_state_dict'    : m.state_dict(),
            'muon_state_dict'     : muon.state_dict(),
            'adamw_state_dict'    : adamw.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        info_path = self.path.replace('.pt', '_info.json')
        info = {**metadata, 'last_save': datetime.now().isoformat(), 'config': CONFIG}
        tmp_json = info_path + '.tmp'
        with open(tmp_json, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        tmp_pt = self.path + '.tmp'
        torch.save(cp, tmp_pt)
        os.replace(tmp_pt, self.path)
        os.replace(tmp_json, info_path)
        print(f'  💾 SAVE  step={metadata["global_step"]:,}  '
              f'chunk_idx={metadata["current_chunk_idx"]}  [{self.path}]')

    def load(self) -> Optional[dict]:
        if not os.path.exists(self.path):
            return None
        print(f'\nCheckpoint trouvé : {self.path}')
        cp        = torch.load(self.path, map_location='cpu', weights_only=False)
        info_path = self.path.replace('.pt', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            # Compatibilité ancien format (epoch/cwi) → nouveau format (chunk_idx)
            if 'current_chunk_idx' not in info:
                old_epoch        = info.get('current_epoch', 1)
                old_cwi          = info.get('chunk_within_epoch', 0)
                chunks_per_epoch = info.get('config', {}).get('chunks_per_epoch', 2)
                reconstructed    = (old_epoch - 1) * chunks_per_epoch + old_cwi
                info['current_chunk_idx'] = reconstructed
                info['chunk_start_step']  = info.get('chunk_start_step', 0)
                print(f'  ⚠️  Ancien format (epoch={old_epoch} cwi={old_cwi}) '
                      f'→ chunk_idx={reconstructed}')
            for k in ('global_step', 'current_chunk_idx',
                      'total_training_time', 'chunk_start_step'):
                cp[k] = info.get(k, 0)
        else:
            cp.update({'global_step': 0, 'current_chunk_idx': 0,
                       'total_training_time': 0.0, 'chunk_start_step': 0})
        return cp


# ── Validation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, val_loader, max_batches: int = 50) -> tuple:
    model.eval()
    total_loss, n = 0.0, 0
    ae  = (DEVICE == 'cuda')
    adt = torch.bfloat16 if ae else torch.float32
    try:
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches: break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(DEVICE, dtype=adt, enabled=ae):
                _, loss, _ = model(x, targets=y)
            total_loss += loss.item(); n += 1
    finally:
        model.train()
    avg = total_loss / max(n, 1)
    return math.exp(min(avg, 10)), avg


# ── Muon + MARS-M ─────────────────────────────────────────────────────────────
def _zeropower_via_newtonschulz5(G, steps: int = 5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * (A @ A); X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                     ns_steps=ns_steps, weight_decay=weight_decay,
                                     use_mars=use_mars, mars_gamma=mars_gamma))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, nest = group['lr'], group['momentum'], group['nesterov']
            ns, wd        = group['ns_steps'], group['weight_decay']
            use_mars, mg  = group.get('use_mars', True), group.get('mars_gamma', 0.025)
            for p in group['params']:
                if p.grad is None or p.grad.ndim < 2: continue
                g     = p.grad
                state = self.state[p]
                if use_mars:
                    if 'prev_grad' not in state:
                        state['prev_grad'] = torch.zeros_like(g)
                    prev = state['prev_grad']
                    c_t  = torch.clamp(
                        (mg / (1. - mg)) * (g.norm() + 1e-8) / (prev.norm() + 1e-8),
                        max=1.0)
                    g = g + c_t * (g - prev)
                    state['prev_grad'].copy_(p.grad)
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(mom).add_(g)
                g   = (g + mom * buf) if nest else buf
                g   = _zeropower_via_newtonschulz5(g, steps=ns)
                g   = g * max(g.size(0), g.size(1)) ** 0.5
                if wd: p.mul_(1. - lr * wd)
                p.add_(g, alpha=-lr)


def configure_optimizers(model, lr: float, weight_decay: float, betas, eps):
    EXCLUDE = {'token_embeddings.weight', 'output_head.weight'}
    muon_params, adamw_decay, adamw_nodecay = [], [], []
    for pn, p in model.named_parameters():
        if not p.requires_grad: continue
        if pn in EXCLUDE:
            (adamw_decay if p.dim() >= 2 else adamw_nodecay).append(p)
        elif p.dim() >= 2 and pn.startswith('blocks.'):
            muon_params.append(p)
        elif p.dim() < 2 and pn.startswith('blocks.'):
            adamw_nodecay.append(p)
        elif p.dim() >= 2:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)

    lr_muon  = lr * 5.0
    muon_opt = Muon(
        [{'params': muon_params, 'is_muon': True}],
        lr=lr_muon, momentum=0.95, nesterov=True,
        ns_steps=3, weight_decay=0.0, use_mars=True, mars_gamma=0.025,
    )
    muon_opt.param_groups[0]['is_muon'] = True
    adamw_opt = torch.optim.AdamW(
        [{'params': adamw_decay,   'weight_decay': weight_decay, 'is_muon': False},
         {'params': adamw_nodecay, 'weight_decay': 0.0,          'is_muon': False}],
        lr=lr, betas=betas, eps=eps, fused=(DEVICE == 'cuda'),
    )
    n_muon  = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_decay + adamw_nodecay)
    print(f'\n  Muon+MARS  : {n_muon / 1e6:.2f}M params  lr={lr_muon:.2e}')
    print(f'  AdamW      : {n_adamw / 1e6:.2f}M params  lr={lr:.2e}')
    return muon_opt, adamw_opt


# ── Train one chunk ───────────────────────────────────────────────────────────
def train_one_chunk(
    model, chunk_info: dict, optimizers, scheduler,
    ckpt_mgr: CheckpointManager, history: dict,
    global_step: int, total_time: float,
    chunk_idx: int, chunk_start_step: int,
    is_resume: bool,
) -> tuple:

    muon_opt, adamw_opt = optimizers

    print(f'\n{"="*70}')
    print(f'  Chunk {chunk_idx}/{len(ALL_CHUNKS)-1} — chunk_{chunk_info["id"]:03d}  '
          f'({chunk_info["tokens"] / 1e9:.2f}B tokens)')
    print(f'{"="*70}')

    chunk    = LazyChunk(chunk_info, CONFIG['max_seq_len'], CONFIG['val_tokens'])
    train_ds = chunk.train_dataset(
        CONFIG['max_seq_len'], CONFIG['use_packing'], tokenizer.eos_token_id)
    val_ds   = chunk.val_dataset(CONFIG['max_seq_len'])

    total_seqs            = len(train_ds)
    steps_done_in_chunk   = global_step - chunk_start_step
    batches_done_in_chunk = steps_done_in_chunk * CONFIG['gradient_accumulation']

    if batches_done_in_chunk >= math.ceil(total_seqs / CONFIG['batch_size']):
        print('  ✅ Chunk déjà traité intégralement — skip')
        chunk.unload()
        return global_step, total_time, chunk_start_step

    rng     = np.random.default_rng(42 + chunk_idx * 1000)
    indices = rng.permutation(total_seqs)
    indices = indices[batches_done_in_chunk * CONFIG['batch_size']:].tolist()

    if is_resume and batches_done_in_chunk > 0:
        print(f'  ↩️  Reprise : {batches_done_in_chunk} batches déjà faits '
              f'({steps_done_in_chunk} steps), {len(indices)} samples restants')

    class IndexSampler(torch.utils.data.Sampler):
        def __init__(self, idx): self._idx = idx
        def __iter__(self): return iter(self._idx)
        def __len__(self):  return len(self._idx)

    collate = (partial(packed_collate_fn,
                       eos_token_id=tokenizer.eos_token_id,
                       seq_len=CONFIG['max_seq_len'])
               if CONFIG['use_packing'] else None)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'],
        sampler=IndexSampler(indices),
        num_workers=CONFIG['num_workers'], pin_memory=True,
        persistent_workers=(CONFIG['num_workers'] > 0),
        prefetch_factor=2 if CONFIG['num_workers'] > 0 else None,
        drop_last=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=1, pin_memory=True,
    )

    total_batches = total_seqs // CONFIG['batch_size']
    print(f'  batches={total_batches:,}  restant={len(train_loader):,}  '
          f'packing={"ON" if CONFIG["use_packing"] else "OFF"}')

    model.train()
    ae            = (DEVICE == 'cuda')
    adt           = torch.bfloat16
    chunk_loss    = 0.0
    valid_batches = 0
    acc_steps     = 0
    t0            = time.time()

    pbar = tqdm(
        train_loader,
        desc=f'  C{chunk_idx}',
        initial=total_batches - len(train_loader),
        total=total_batches, leave=True, dynamic_ncols=True,
    )

    for batch in pbar:
        try:
            if CONFIG['use_packing'] and len(batch) == 4:
                x, y, cu_seqlens, max_sl = batch
                x          = x.to(DEVICE, non_blocking=True)
                y          = y.to(DEVICE, non_blocking=True)
                cu_seqlens = cu_seqlens.to(DEVICE, non_blocking=True)
            else:
                x, y       = batch[0].to(DEVICE), batch[1].to(DEVICE)
                cu_seqlens = max_sl = None

            with torch.amp.autocast(DEVICE, dtype=adt, enabled=(DEVICE == 'cuda')):
                _, loss, _ = model(
                    x, targets=y,
                    cu_seqlens_q = cu_seqlens,
                    cu_seqlens_k = cu_seqlens,
                    max_seqlen_q = int(max_sl) if max_sl is not None else None,
                    max_seqlen_k = int(max_sl) if max_sl is not None else None,
                )
                loss = loss / CONFIG['gradient_accumulation']

            if torch.isnan(loss) or torch.isinf(loss):
                acc_steps = 0
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                continue

            loss.backward()
            chunk_loss    += loss.item() * CONFIG['gradient_accumulation']
            valid_batches += 1
            acc_steps     += 1

            if acc_steps >= CONFIG['gradient_accumulation']:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG['max_grad_norm'])
                muon_opt.step()
                adamw_opt.step()
                lr = scheduler.step()
                muon_opt.zero_grad(set_to_none=True)
                adamw_opt.zero_grad(set_to_none=True)
                acc_steps   = 0
                global_step += 1

                avg_loss = chunk_loss / max(valid_batches, 1)
                pbar.set_postfix(
                    loss=f'{avg_loss:.4f}',
                    ppl =f'{math.exp(min(avg_loss, 10)):.1f}',
                    lr  =f'{lr:.2e}',
                )

                # Validation
                if global_step % CONFIG['validate_every_steps'] == 0:
                    ppl, vloss = validate(model, val_loader, CONFIG['val_batches'])
                    pbar.write(f'  [val  step={global_step:,}] '
                               f'loss={vloss:.4f}  ppl={ppl:.2f}')
                    history.setdefault('validations', []).append({
                        'step': global_step, 'val_loss': vloss, 'val_ppl': ppl})

                # Checkpoint périodique
                if global_step % CONFIG['save_every_steps'] == 0:
                    ckpt_mgr.save(model, optimizers, scheduler, {
                        'global_step'        : global_step,
                        'current_chunk_idx'  : chunk_idx,
                        'total_training_time': total_time + (time.time() - t0),
                        'chunk_start_step'   : chunk_start_step,
                    })

                # Timed save + push HF
                timed_hf_save(ckpt_mgr, model, optimizers, scheduler, {
                    'global_step'        : global_step,
                    'current_chunk_idx'  : chunk_idx,
                    'total_training_time': total_time + (time.time() - t0),
                    'chunk_start_step'   : chunk_start_step,
                })

                # Affichage graph_scale (signal Naylis)
                if global_step % 1000 == 0:
                    raw    = model._orig_mod if hasattr(model, '_orig_mod') else model
                    scales = [b.attention.graph_scale.detach().abs().mean().item()
                              for b in raw.blocks]
                    avg_s  = sum(scales) / len(scales)
                    g_min  = min(scales); g_max = max(scales)
                    pbar.write(f'  [naylis step={global_step:,}] '
                               f'|graph_scale| avg={avg_s:.5f}  '
                               f'min={g_min:.5f}  max={g_max:.5f}')

        except torch.cuda.OutOfMemoryError:
            print(f'\n  OOM — skip batch')
            torch.cuda.empty_cache()
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)
            acc_steps = 0
            gc.collect()
            model.train()
            continue

    pbar.close()
    elapsed     = time.time() - t0
    total_time += elapsed
    avg_loss    = chunk_loss / max(valid_batches, 1)
    print(f'\n  chunk_{chunk_info["id"]:03d} terminé | '
          f'loss={avg_loss:.4f} | ppl={math.exp(min(avg_loss, 10)):.2f} | '
          f'{elapsed / 60:.1f}min')

    history.setdefault('chunks', []).append({
        'chunk_idx': chunk_idx, 'chunk_id': chunk_info['id'],
        'loss': avg_loss, 'time_sec': elapsed, 'global_step': global_step,
    })

    chunk.unload()
    del train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_step, total_time, chunk_start_step


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '=' * 70)
    print('  CRÉATION MODÈLE')
    print('=' * 70)

    ckpt_mgr = CheckpointManager(CONFIG['checkpoint_file'])

    model = NaylisGPT(
        vocab_size            = CONFIG['vocab_size'],
        embed_dim             = CONFIG['embed_dim'],
        num_heads             = CONFIG['num_heads'],
        num_layers            = CONFIG['num_layers'],
        max_seq_len           = CONFIG['max_seq_len'],
        dropout               = CONFIG['dropout'],
        use_rope              = CONFIG['use_rope'],
        use_yarn              = CONFIG['use_yarn'],
        yarn_scale            = CONFIG['yarn_scale'],
        yarn_original_max_len = CONFIG['yarn_original_max_len'],
        use_swiglu            = CONFIG['use_swiglu'],
        n_kv_heads            = CONFIG['n_kv_heads'],
        use_qk_norm           = CONFIG['use_qk_norm'],
        soft_cap              = CONFIG['soft_cap'],
        use_flash_attn        = CONFIG['use_flash_attn'],
        rel_rank              = CONFIG['rel_rank'],
    ).to(DEVICE)

    p = model.count_parameters()
    print(f'  Params total : {p["total_M"]}M')
    print(f'  Naylis       : {p["naylis_K"]}K = {p["naylis_pct"]}')

    # torch.compile
    if CONFIG['use_compile'] and DEVICE == 'cuda':
        print('\ntorch.compile...')
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors  = True
        try:
            model = torch.compile(model, mode=CONFIG['compile_mode'])
            print('  OK')
        except Exception as e:
            print(f'  FAIL : {e}')
    else:
        print('\ntorch.compile : désactivé')

    raw_model  = model._orig_mod if hasattr(model, '_orig_mod') else model
    optimizers = configure_optimizers(
        raw_model, CONFIG['learning_rate'], CONFIG['weight_decay'],
        (CONFIG['adam_beta1'], CONFIG['adam_beta2']), CONFIG['adam_eps'],
    )
    muon_opt, adamw_opt = optimizers

    sched_total_steps = ARGS.total_steps if ARGS.total_steps is not None else TOTAL_STEPS
    if ARGS.total_steps is not None:
        print(f'\n  ⚠️  --total-steps forcé : {sched_total_steps:,} '
              f'(calculé={TOTAL_STEPS:,})')
    scheduler = WSDScheduler(
        list(optimizers), max_lr=CONFIG['learning_rate'],
        total_steps=sched_total_steps, warmup_ratio=CONFIG['warmup_ratio'],
        decay_ratio=CONFIG['decay_ratio'], min_lr_ratio=CONFIG['min_lr_ratio'],
    )

    history           = {'config': CONFIG, 'chunks': [], 'validations': []}
    global_step       = 0
    current_chunk_idx = 0
    total_time        = 0.0
    chunk_start_step  = 0
    cp                = None

    cp = ckpt_mgr.load()
    if cp:
        print('\nREPRISE')
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(cp['model_state_dict'])
        muon_opt.load_state_dict(cp.get('muon_state_dict', {}))
        adamw_opt.load_state_dict(cp.get('adamw_state_dict', {}))
        scheduler.load_state_dict(cp.get('scheduler_state_dict', {}))
        global_step       = cp.get('global_step', 0)
        current_chunk_idx = cp.get('current_chunk_idx', 0)
        total_time        = cp.get('total_training_time', 0.0)
        chunk_start_step  = cp.get('chunk_start_step', 0)
        print(f'  global_step={global_step:,}  chunk_idx={current_chunk_idx}  '
              f'total_time={total_time/3600:.2f}h')

        if current_chunk_idx >= len(ALL_CHUNKS):
            print('\n✅ Training déjà terminé sur tous les chunks disponibles.')
            return

    print('\n' + '=' * 70)
    print(f'  TRAINING START — {TOTAL_STEPS:,} steps estimés — {len(ALL_CHUNKS)} chunks')
    print('=' * 70)

    # ── Boucle linéaire sur les chunks ───────────────────────────────────────
    for chunk_idx, chunk_info in enumerate(ALL_CHUNKS):

        if chunk_idx < current_chunk_idx:
            print(f'  ⏩ chunk_{chunk_info["id"]:03d} (idx={chunk_idx}) — déjà traité, skip')
            continue

        is_resume        = (cp is not None and chunk_idx == current_chunk_idx)
        chunk_start_step = chunk_start_step if is_resume else global_step

        try:
            global_step, total_time, chunk_start_step = train_one_chunk(
                model            = model,
                chunk_info       = chunk_info,
                optimizers       = optimizers,
                scheduler        = scheduler,
                ckpt_mgr         = ckpt_mgr,
                history          = history,
                global_step      = global_step,
                total_time       = total_time,
                chunk_idx        = chunk_idx,
                chunk_start_step = chunk_start_step,
                is_resume        = is_resume,
            )
            # Chunk terminé → save + avancer l'index
            current_chunk_idx = chunk_idx + 1
            ckpt_mgr.save(model, optimizers, scheduler, {
                'global_step'        : global_step,
                'current_chunk_idx'  : current_chunk_idx,
                'total_training_time': total_time,
                'chunk_start_step'   : global_step,
            })
            cp = None

        except KeyboardInterrupt:
            print('\n  CTRL+C — sauvegarde d\'urgence...')
            ckpt_mgr.save(model, optimizers, scheduler, {
                'global_step'        : global_step,
                'current_chunk_idx'  : chunk_idx,
                'total_training_time': total_time,
                'chunk_start_step'   : chunk_start_step,
            })
            print('  ✅ Sauvegarde OK')
            print('  ☁️  Push HF urgence...')
            hf_push_checkpoint(CONFIG['checkpoint_file'], global_step, push_cache=True)
            return

        except Exception:
            print(f'\n  ERREUR :\n{traceback.format_exc()}')
            ckpt_mgr.save(model, optimizers, scheduler, {
                'global_step'        : global_step,
                'current_chunk_idx'  : chunk_idx,
                'total_training_time': total_time,
                'chunk_start_step'   : chunk_start_step,
            })
            raise

    # ── Fin du training ───────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  ✅ TRAINING TERMINÉ — tous les chunks traités')
    print(f'{"="*70}')
    print(f'  Steps  : {global_step:,}')
    print(f'  Temps  : {total_time / 3600:.2f}h')
    print(f'  Chunks : {len(ALL_CHUNKS)}')

    ckpt_mgr.save(model, optimizers, scheduler, {
        'global_step'        : global_step,
        'current_chunk_idx'  : len(ALL_CHUNKS),
        'total_training_time': total_time,
        'chunk_start_step'   : global_step,
    })

    hist_path = CONFIG['checkpoint_file'].replace('.pt', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f'  History : {hist_path}')

    print('\n  ☁️  Push final sur HF...')
    hf_push_checkpoint(CONFIG['checkpoint_file'], global_step, push_cache=True)
    print('  DONE')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrompu')
    except Exception:
        print(traceback.format_exc())
    finally:
        print('\nBye')

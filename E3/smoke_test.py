from __future__ import annotations

import functools
import contextlib
import csv
import io
import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader, TensorDataset

from E1.models import build_model as build_e1_model
from E2.models import build_model as build_e2_model
from E3.models import (SEQUENCE_EXPERIMENTS, CNNStemSequenceTokenizer,
                       PatchSequenceTokenizer, RowSequenceTokenizer, build_image_sequence_model)
from E3.train import build_experiment_model, parse_args, reset_loaders, run_experiments
from E3.baselines import DEFAULT_BASELINES, load_e1_baselines
from common.engine import fit_classifier
from common.data import DataBundle
from common.utils import set_seed
from E3.text_data import (
    TOY_SENTIMENT_SAMPLES,
    SentimentDataset,
    build_vocab,
    collate_sentiment_batch,
)
from E3.text_models import SentimentRNNClassifier


def assert_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")
    print(f"OK {name}: {actual}")


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    images = torch.randn(4, 1, 28, 28)

    for experiment in SEQUENCE_EXPERIMENTS:
        model = build_image_sequence_model(experiment=experiment)
        logits = model(images)
        assert_shape(experiment, tuple(logits.shape), (4, 10))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0,1,2,3])).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    # Coordinate-coded pixels detect incorrect flattening, scanning and dropped borders.
    coordinates = torch.arange(784).float().view(1,1,28,28)
    torch.testing.assert_close(RowSequenceTokenizer()(coordinates), coordinates[0,0].unsqueeze(0))
    for stride in (7,3):
        tokenizer = PatchSequenceTokenizer(7,stride)
        expected = torch.stack([coordinates[0,0,y:y+7,x:x+7].reshape(-1)
                                for y in range(0,22,stride) for x in range(0,22,stride)]).unsqueeze(0)
        torch.testing.assert_close(tokenizer(coordinates),expected)
        assert expected[0,-1,-1].item()==783
    stem = CNNStemSequenceTokenizer(8)
    torch.testing.assert_close(stem(images),stem.stem(images).flatten(2).transpose(1,2))
    assert_shape('cnn_stem_tokens',tuple(stem(images).shape),(4,49,8))
    for kwargs in ({'patch_size':6},{'patch_size':7,'stride':4},{'patch_size':0}):
        try:
            PatchSequenceTokenizer(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid patch geometry accepted: {kwargs}')
    for family in ('lstm','gru'):
        model = build_image_sequence_model(f'{family}_row',num_layers=2,bidirectional=True,dropout=0)
        _,state = model.rnn(model.tokenizer(images))
        hidden = state[0] if family=='lstm' else state
        torch.testing.assert_close(model(images),model.head(torch.cat((hidden[-2],hidden[-1]),1)))

    for baseline in ["mlp", "cnn"]:
        model = build_e1_model(baseline)
        logits = model(images)
        assert_shape(f"baseline_{baseline}", tuple(logits.shape), (4, 10))

    for experiment in ["custom_patch", "pytorch_patch", "custom_row"]:
        model = build_e2_model(experiment=experiment)
        logits = model(images)
        assert_shape(f"e2_{experiment}", tuple(logits.shape), (4, 10))

    vocab = build_vocab(TOY_SENTIMENT_SAMPLES)
    dataset = SentimentDataset(TOY_SENTIMENT_SAMPLES[:6], vocab)
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=functools.partial(collate_sentiment_batch, pad_index=vocab.pad_index),
    )
    tokens, lengths, labels = next(iter(loader))
    assert_shape("sentiment_tokens", tuple(tokens.shape), (4, int(lengths.max().item())))
    assert_shape("sentiment_lengths", tuple(lengths.shape), (4,))
    assert_shape("sentiment_labels", tuple(labels.shape), (4,))

    for rnn_type in ["lstm", "gru"]:
        model = SentimentRNNClassifier(
            vocab_size=len(vocab.itos),
            pad_index=vocab.pad_index,
            rnn_type=rnn_type,
            bidirectional=True,
        )
        logits = model(tokens, lengths)
        assert_shape(f"sentiment_{rnn_type}", tuple(logits.shape), (4, 2))

    # Full pipeline test: train, choose best val epoch, save/load checkpoints,
    # predictions/CM, comparisons, optional order probe and overwrite protection.
    test_pipeline()
    print("All E3 smoke tests passed.")


def test_pipeline() -> None:
    cache = ROOT_DIR / '.matplotlib-cache'
    cache.mkdir(exist_ok=True)
    temporary = cache / ('e3_smoke_' + uuid.uuid4().hex)
    temporary.mkdir()
    assert temporary.resolve().is_relative_to(cache.resolve())
    try:
        # Test fixtures are clearly synthetic; never mix real MNIST metrics
        # into a run trained on random images.
        fixture = temporary / 'e1_fixture'
        fixture.mkdir()
        fixture_metrics = {
            'dataset': 'synthetic_test',
            'split': {'train': 10, 'validation': 10, 'test': 10, 'seed': 42, 'val_ratio': .1},
            'training': {'epochs': 2, 'batch_size': 4, 'learning_rate': .001, 'weight_decay': .0001},
            'models': {}
        }
        for name, count in [('mlp', 235146), ('cnn', 421642)]:
            fixture_metrics['models'][name] = {
                'model': name, 'parameters': count, 'best_epoch': 1, 'best_val_accuracy': .5,
                'test_accuracy': .1, 'macro_f1': .1, 'test_loss': 2.3,
                'training_time': '1s', 'training_time_seconds': 1.0
            }
            (fixture/f'history_{name}.csv').write_text(
                'epoch,train_loss,train_accuracy,val_loss,val_accuracy\n'
                '1,2.3,0.1,2.3,0.5\n2,2.2,0.2,2.4,0.4\n', encoding='utf-8')
        fixture_path = fixture / 'metrics.json'
        fixture_path.write_text(json.dumps(fixture_metrics), encoding='utf-8')
        fixture_before = fixture_path.read_bytes()
        args = parse_args(['--epochs','2','--batch-size','4','--num-workers','0','--hidden-dim','8',
                           '--stem-dim','8','--device','cpu','--probe-order','--baseline-results',str(fixture),
                           '--output-dir',str(Path(temporary)/'run')])
        images = torch.randn(10,1,28,28)
        labels = torch.arange(10)
        dataset = TensorDataset(images,labels)
        loader = DataLoader(dataset,batch_size=4)
        data = DataBundle(loader,loader,loader,10,10,10,10,(1,28,28))
        # The shuffle schedule stays the same despite backend RNG consumption.
        left = reset_loaders(data,args).train_loader
        right = reset_loaders(data,args).train_loader
        for _ in range(2):
            a = torch.cat([y for _,y in left]); torch.randn(103)
            b = torch.cat([y for _,y in right]); torch.testing.assert_close(a,b)
        # CNN-stem initialization is shared before different recurrent modules.
        set_seed(42); a = build_experiment_model('lstm_cnn_stem',args)
        set_seed(42); b = build_experiment_model('gru_cnn_stem',args)
        for key,value in a.tokenizer.state_dict().items():
            torch.testing.assert_close(value,b.tokenizer.state_dict()[key],rtol=0,atol=0)
        from unittest.mock import patch
        with patch('E3.train.fit_classifier', wraps=fit_classifier) as trainer:
            with contextlib.redirect_stdout(io.StringIO()):
                payload = run_experiments(args,data)
            assert trainer.call_count == 8
            assert all(call.args[0].rnn_type in ('lstm','gru') for call in trainer.call_args_list)
        assert payload['status']=='completed' and payload['dataset']=='synthetic_test'
        assert len(payload['experiments'])==8
        assert fixture_path.read_bytes() == fixture_before
        assert set(payload['baselines']) == {'mlp_baseline','cnn_baseline'}
        for baseline in payload['baselines'].values():
            assert baseline['source'] == 'E1_reused' and not baseline['trained_in_this_run']
            assert baseline['test_accuracy'] == .1
        assert not list((args.output_dir/'checkpoints').glob('*baseline*'))
        assert len(list(csv.DictReader((args.output_dir/'summary.csv').open()))) == 8
        comparison = list(csv.DictReader((args.output_dir/'comparison_baselines.csv').open()))
        assert len(comparison) == 10
        assert sum(r['source'] == 'E1_reused' and r['trained_in_this_run'] == 'False' for r in comparison) == 2
        for name,metric in payload['experiments'].items():
            history = list(csv.DictReader((args.output_dir/f'history_{name}.csv').open()))
            assert len(history)==2
            best = max(history,key=lambda r:float(r['val_accuracy']))
            assert int(float(best['epoch']))==metric['best_epoch']
            cp = torch.load(args.output_dir/metric['best_checkpoint'],map_location='cpu',weights_only=True)
            model = build_experiment_model(name,args)
            model.load_state_dict(cp['state_dict'],strict=True); model.eval()
            with torch.no_grad():
                predictions = model(images).argmax(1).tolist()
            recorded = list(csv.DictReader((args.output_dir/f'predictions_{name}.csv').open()))
            assert predictions==[int(r['prediction']) for r in recorded]
            matrix = list(csv.DictReader((args.output_dir/f'confusion_matrix_{name}.csv').open()))
            assert sum(int(r[f'pred_{j}']) for r in matrix for j in range(10))==10
            accuracy = sum(p==int(t) for p,t in zip(predictions,labels))/10
            assert accuracy==metric['test_accuracy']
            if name in SEQUENCE_EXPERIMENTS:
                probe = metric['order_probe']
                assert probe['split']=='validation'
                assert probe['orders']['raster']['accuracy']==metric['best_val_accuracy']
                assert all(sorted(o['permutation'])==list(range(metric['sequence_length'])) for o in probe['orders'].values())
        rows = list(csv.DictReader((args.output_dir/'comparison_rnn.csv').open()))
        assert len(rows)==4 and all(r['pair_complete']=='True' for r in rows)
        assert len(list(csv.DictReader((args.output_dir/'comparison_order_probe.csv').open())))==24
        before = (args.output_dir/'metrics.json').read_bytes()
        try:
            run_experiments(args,data)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing output was overwritten')
        assert (args.output_dir/'metrics.json').read_bytes()==before
        assert not (args.output_dir/'.run.lock').exists()
        # A failed run preserves metadata and releases only its own lock.
        failed = parse_args(['--experiments','lstm_row','--epochs','2','--batch-size','4','--device','cpu',
                             '--baseline-results',str(fixture),'--num-workers','0',
                             '--output-dir',str(Path(temporary)/'failed')])
        with patch('E3.train.fit_classifier',side_effect=RuntimeError('intentional failure')):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_experiments(failed,data)
            except RuntimeError:
                pass
        assert json.loads((failed.output_dir/'metrics.json').read_text(encoding='utf-8'))['status']=='failed'
        assert not (failed.output_dir/'.run.lock').exists()
        for option, value in [('seed', 43), ('epochs', 100), ('batch_size', 256)]:
            previous = getattr(args, option)
            setattr(args, option, value)
            try:
                load_e1_baselines(args, 'synthetic_test')
            except ValueError:
                pass
            else:
                raise AssertionError(f'Incompatible E1 baseline {option} accepted')
            setattr(args, option, previous)
        # Real E1 values used by the report remain byte-for-byte unchanged.
        real_path = DEFAULT_BASELINES / 'metrics.json'
        real_before = real_path.read_bytes()
        real, _histories = load_e1_baselines(parse_args([]))
        assert real['mlp_baseline']['test_accuracy'] == .9863
        assert real['cnn_baseline']['test_accuracy'] == .9927
        assert real_path.read_bytes() == real_before
    finally:
        assert temporary.resolve().is_relative_to(cache.resolve())
        shutil.rmtree(temporary)
    print('OK full pipeline: 8 RNN training calls, 2 reused E1 baselines, provenance and protocol checks')


if __name__ == "__main__":
    main()


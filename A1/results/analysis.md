# A1.2 Results Analysis

## Best Results

- Best test accuracy: `vit_b16_pretrained_partial` with 0.8325.
- Best macro F1: `vit_b16_pretrained_partial` with 0.8324.
- CNN average test accuracy: 0.6764.
- Transformer average test accuracy: 0.6963.

## Summary Table

| Experiment | Family | Mode | Trainable params | Best val acc | Test acc | Macro F1 | Time |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| resnet18_scratch | CNN | from_scratch | 11,227,812 | 0.6164 | 0.6097 | 0.6070 | 2h 27m 17s |
| resnet18_pretrained_head | CNN | pretrained_freeze_backbone_train_head | 51,300 | 0.6108 | 0.6047 | 0.6022 | 1h 24m 4s |
| resnet18_pretrained_partial | CNN | pretrained_freeze_partial | 8,445,028 | 0.7283 | 0.7227 | 0.7225 | 1h 33m 4s |
| resnet18_pretrained_full | CNN | pretrained_full_finetune | 11,227,812 | 0.7729 | 0.7684 | 0.7685 | 2h 25m 35s |
| vit_b16_scratch | Transformer | from_scratch | 85,875,556 | 0.3537 | 0.3530 | 0.3426 | 18h 23m 37s |
| vit_b16_pretrained_head | Transformer | pretrained_freeze_backbone_train_head | 76,900 | 0.8218 | 0.8132 | 0.8130 | 26h 8m 14s |
| vit_b16_pretrained_partial | Transformer | pretrained_freeze_partial | 14,252,644 | 0.8366 | 0.8325 | 0.8324 | 28h 48m 53s |
| vit_b16_pretrained_full | Transformer | pretrained_full_finetune | 85,875,556 | 0.7873 | 0.7864 | 0.7859 | 40h 44m 46s |

## Data Provenance and Limits

Restored from user-supplied console logs: `resnet18_scratch`, `resnet18_pretrained_head`, `resnet18_pretrained_full`, `vit_b16_pretrained_head`, `vit_b16_pretrained_partial`, `vit_b16_pretrained_full`.
These test metrics have four decimal places; durations are reported to whole seconds. Their parameter counts are expected counts derived from the current model definition, rather than recovered measurements. Their full runtime configurations still need confirmation.
The user reported concurrent execution. GPU models and resource allocation are not recorded, so elapsed durations are not a controlled speed comparison.
Family averages above combine different training modes; they are not averages over repeated seeds.
See `README.md`, `metrics.json` and `results_inventory.csv` for the complete provenance.
- `vit_b16_scratch`: nonfinite loss from epoch 11; best validation state at epoch 10. Do not interpret this as stable training throughout the recorded history.
- `vit_b16_pretrained_full`: nonfinite loss from epoch 37; best validation state at epoch 3. Do not interpret this as stable training throughout the recorded history.

## Interpretation Prompts

- Compare from-scratch vs pretrained within ResNet18 and ViT-B/16.
- Check whether head-only freeze is fast but less adaptive than partial/full fine-tuning.
- Discuss whether ViT-B/16 benefits more from pretrained weights than ResNet18 on full CIFAR-100.
- Use confusion matrices and misclassified samples to name classes that are frequently confused.

## Generated Figures

- `bar_test_accuracy.png`
- `bar_macro_f1.png`
- `bar_trainable_parameters.png`
- `bar_training_time_seconds.png`

# Fine-tuning Inception v3 for GTZAN

This document explains how `src/cnn.py` adapts torchvision's Inception v3
for fine-tuning on the GTZAN music genre dataset.

## What the class represents

`InceptionV3` is a thin wrapper around `torchvision.models.inception_v3`
that performs three things:

1. Loads the ImageNet-pretrained weights.
2. Replaces the final classification layer with a 10-way head
   (one output per GTZAN genre).
3. Freezes all original parameters and trains only the new head.

It is the network that `fine_tuning.py` instantiates and trains.

## The architecture

Inception v3 (and most image classifiers) is structured as:

```
[convolutional layers]  →  [global avg pooling]  →  [fc layer]  →  [logits]
   feature extraction        collapse to vector         classification
```

In torchvision's implementation, the final fully connected layer is
exposed as `model.fc`.

## What `fc` means and why `nn.Linear`

`fc` stands for **"fully connected"** — a layer where every input
neuron is connected to every output neuron. In PyTorch this is
implemented by `nn.Linear(in_features, out_features)`.

The original `model.fc` is `nn.Linear(2048, 1000)`, trained on
ImageNet's 1000 classes. For GTZAN we replace it:

```python
self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
```

`in_features` (2048) stays the same — that's the size of the pooled
feature vector coming out of the convolutional stack. Only `out_features`
changes from 1000 to 10.

## Why the freeze pattern

```python
if freeze_backbone:
    for param in self.backbone.parameters():
        param.requires_grad = False
    for param in self.backbone.fc.parameters():
        param.requires_grad = True
```

`requires_grad` is PyTorch's per-parameter flag that tells autograd
whether to compute gradients during backpropagation. `requires_grad=False`
means "don't compute gradients" — which in turn means the optimizer
won't update that parameter.

The two-loop pattern does the following:

- **First loop** freezes every parameter in the entire backbone
  (convolutions, batch norms, the auxiliary head if any, the original
  `fc`). They keep their pretrained ImageNet weights forever.
- **Second loop** re-enables `requires_grad` on just the new `fc` we
  just created, so only the classification head gets trained.

The two loops are necessary because the first loop froze everything
**including** the new `fc` (it was assigned via
`self.backbone.fc = nn.Linear(...)` before step 1), and we need to
selectively unfreeze just the head.

### Why freeze the backbone at all?

For fine-tuning on a small dataset (GTZAN has only 1000 songs), freezing
the backbone gives:

1. **Leverage pretrained features.** ImageNet-trained convolutions
   already detect edges, textures, shapes — useful features for any
   image task, including spectrograms.
2. **Train very few parameters.** Only the ~20k parameters of the new
   `fc` train. Faster, less data needed.
3. **Avoid catastrophic forgetting.** Without freezing, the optimizer
   would rewrite the pretrained weights on top of GTZAN's 1000 songs,
   throwing away the ImageNet knowledge.
4. **Reduce overfitting.** A frozen backbone + tiny trainable head is
   far less prone to overfitting on a small dataset.

## The torchvision quirk with `aux_logits`

Inception v3 has an auxiliary classifier (`model.AuxLogits`) attached
to an intermediate layer, used during the original ImageNet training as:

1. **Vanishing-gradient mitigation** — pushes a second loss signal
   back into the early layers of the deep stack.
2. **Regularization** — encourages discriminative features at multiple
   depths.

Both benefits are **useless for our fine-tuning setup**:

- The early layers the aux loss is meant to push gradients into are
  **frozen** — there is no gradient to push through them.
- Regularization on a frozen feature extractor doesn't help.
- The aux head itself is frozen (per our freeze rule), so it can't
  adapt to be useful even if it could.

### The trap

When loading pretrained weights, torchvision's `inception_v3` builder
unconditionally sets `aux_logits=True` because the saved checkpoint
includes the aux head. Passing `aux_logits=False` together with
`weights=...` raises:

```
ValueError: The parameter 'aux_logits' expected value True but got False instead.
```

There is no way to disable the aux head at load time when using
pretrained weights.

## How we handle aux logits

Two small adjustments in `src/cnn.py` reconcile the constraint with
our "single-tensor forward" preference:

```python
self.backbone = models.inception_v3(weights=weights, aux_logits=True)
self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
self.backbone.aux_logits = False
```

1. **Load with `aux_logits=True`** — required by torchvision.
2. **Set `self.backbone.aux_logits = False`** — flips a flag that the
   forward pass checks before invoking the aux branch. The branch is
   silently skipped.

The pretrained `AuxLogits` head (a 1000-class linear layer) is left
untouched. It's frozen by the standard freeze loop and never called
in forward, so its 1000-class output is irrelevant.

The `forward` override then hides the always-`None` aux from callers:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.backbone(x)[0]
```

`self.backbone(x)` returns `(main_logits, None)`. Indexing `[0]` keeps
just the main logits, so callers see a single tensor:

```python
logits = model(images)                # [B, 10], no tuple to unpack
loss = criterion(logits, labels)
```

## What does NOT change

- The `requires_grad` freeze logic — it already iterates
  `self.backbone.parameters()`, which includes `AuxLogits` and its
  `fc`. The aux head stays frozen.
- `trainable_parameters()` and `parameter_summary()` — they filter
  by `requires_grad`, so they correctly identify only the main `fc`
  as trainable.

## Summary table

| Element | What it does |
|---|---|
| `fc` | The final fully connected classification layer |
| `nn.Linear` | PyTorch's implementation of a fully connected layer |
| `requires_grad = False` | Tells autograd to skip this parameter |
| `requires_grad = True` | Re-enables a parameter for training |
| `aux_logits` (torchvision flag) | Whether to include the auxiliary classifier branch |
| `aux_logits=True` (forced) | Required by torchvision when loading pretrained weights |
| `aux_logits=False` (after load) | Disables the aux branch in the forward pass |
| `self.backbone(x)[0]` | Drops the always-`None` aux from the output |

## Training loop shape

Because of the `forward` override, the training loop is the standard
single-output pattern:

```python
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

No tuple unpacking, no `0.4 * aux_loss` term, no aux head management.
The fine-tuning loop reads exactly like any other classification loop.
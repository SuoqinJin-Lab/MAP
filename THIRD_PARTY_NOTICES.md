# Third-party notices

## CRISP PertAE

The CRISP baseline adapts the `PertAE` architecture and preprocessing defaults
from [ml4bio/CRISP](https://github.com/ml4bio/CRISP), commit
`512d04f95b10e780fef3245825884cb47e83e288` distributed in `CRISP-main.zip`.

Copyright (c) 2025 Xinyuan LIU. Licensed under the MIT License.

## XPert

The XPert baseline adapts the dual-branch attention topology and UniMol token
layout from [GSanShui/XPert](https://github.com/GSanShui/XPert).

Copyright (c) 2025 Guo Yue. Licensed under the MIT License.

## Conditional Monge Gap

The CMonge baseline adapts the architecture, conditioning scheme, and Monge
Gap objective from [AI4SCR/Conditional-Monge](https://github.com/AI4SCR/Conditional-Monge)
and the [Nature Machine Intelligence article](https://doi.org/10.1038/s42256-026-01242-8).

The upstream implementation is licensed under the MIT License. This repository
uses an independent PyTorch and GeomLoss implementation so the baseline can run
in the shared MAP environment without the upstream JAX/OTT dependency stack.

For each MIT-licensed work above, permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following
conditions:

The applicable copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

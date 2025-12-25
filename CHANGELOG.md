<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

Release Notes
=============

1.0.1 (2025-12-18)
------------------
### Bug Fixes
- Fix a bug in hash function that resulted in potential performance regression
    for kernels with many specializations.
- Fix a bug where an if statement within a loop can trigger an internal compiler error.
- Fix SliceType `__eq__` comparison logic.

### Enhancements
- Improve error message for `ct.cat()`.
- Support `is not None` comparison.


1.0.0 (2025-12-02)
------------------
Initial release.

# Class Mapping

| Source class | Decision | Rationale |
| --- | --- | --- |
| `vape`, `Vape`, `ecigarette`, `e-cigarette` | `vape` positive | Confirmed vape synonym. |
| `cigarette` | hard negative | Traditional cigarette is not a vape. |
| `nonVape`, `notVape` | hard negative | Explicit negative class remains negative. |
| `possiblyVape` | manual review | Ambiguous class is not promoted automatically. |
| `Cartridge`, `Battery` | manual review | Component is not assumed to be a complete device. |
| unknown classes | manual review | No unverified class is promoted to positive. |

All curated records retain source class names and raw relative paths. Box-only records are marked `mask_pending_manual_annotation`; no masks are synthesized.

# Data Setup

The experiments use the WhoIsWho from-scratch name disambiguation data. The
dataset is not redistributed in this repository. Place locally obtained files
under:

```text
data/whoiswho/data/
├── NA_Demo/SND/valid/
│   ├── sna_valid_raw.json
│   ├── sna_valid_pub.json
│   └── sna_valid_ground_truth.json
└── v3/SND/valid/
    ├── sna_valid_raw.json
    ├── sna_valid_pub.json
    └── sna_valid_example.json
```

`sna_valid_raw.json` maps each ambiguous name to its paper IDs,
`sna_valid_pub.json` stores publication metadata, and the ground-truth/example
file maps latent real-author IDs to their papers.

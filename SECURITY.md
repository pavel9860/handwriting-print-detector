# Security

Report vulnerabilities privately via [GitHub security advisories](https://github.com/pavel9860/handwriting-print-detector/security/advisories/new),
not in public issues. Weights are loaded with `torch.load` through Ultralytics: load only files whose sha256 matches
`models/registry.json` (the registry checks this) or that you trained yourself.

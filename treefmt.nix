{
  projectRootFile = "flake.nix";

  settings.excludes = [
    "tests/golden/**"
    "tests/fixtures/**"
    "LICENSES/**"
    "NOTICE"
    "uv.lock"
    ".artifacts/**"
    "reports/**"
    "docs/local/**"
  ];

  programs = {
    actionlint.enable = true; # .github/workflows/*.yaml
    deadnix.enable = true;
    statix.enable = true;
    # treefmt orders by priority, then name: deadnix -> statix -> nixfmt.
    nixfmt = {
      enable = true;
      priority = 1;
    };

    taplo.enable = true; # *.toml; discovers taplo.toml at the repo root
    typos.enable = true; # reads [tool.typos] from pyproject.toml
    rumdl-check.enable = true; # discovers .rumdl.toml at the repo root
    ruff-check.enable = true; # reads [tool.ruff] from pyproject.toml
    ruff-format.enable = true;
  };
}

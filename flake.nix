{
  description = "Native MLX engines and Docling stage adapters for Apple Silicon";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    git-hooks-nix = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      git-hooks-nix,
    }:
    let
      eachSystem =
        f:
        nixpkgs.lib.genAttrs [
          "aarch64-darwin"
          "aarch64-linux"
          "x86_64-darwin"
          "x86_64-linux"
        ] (system: f nixpkgs.legacyPackages.${system});

      treefmtFor = pkgs: (treefmt-nix.lib.evalModule pkgs ./treefmt.nix).config;
    in
    {
      formatter = eachSystem (pkgs: (treefmtFor pkgs).build.wrapper);

      checks = eachSystem (
        pkgs:
        let
          treefmt = treefmtFor pkgs;
        in
        {
          # Runs the same hooks `direnv allow` installs locally, treefmt included.
          pre-commit = git-hooks-nix.lib.${pkgs.stdenv.hostPlatform.system}.run {
            src = ./.;
            hooks = {
              treefmt = {
                enable = true;
                packageOverrides.treefmt = treefmt.build.wrapper;
              };
              # `statix check`: treefmt only runs `statix fix`, which skips unfixable lints.
              statix.enable = true;
              gitleaks = {
                enable = true;
                entry = "${pkgs.gitleaks}/bin/gitleaks git --pre-commit --redact --staged";
                pass_filenames = false;
              };
              committed = {
                enable = true;
                entry = "${pkgs.committed}/bin/committed --fixup --wip --commit-file";
                stages = [ "commit-msg" ];
              };
            };
          };

          reuse = pkgs.runCommandLocal "reuse-lint" { } ''
            ${pkgs.reuse}/bin/reuse --root ${self} lint && touch $out
          '';
        }
      );

      devShells = eachSystem (
        pkgs:
        let
          treefmt = treefmtFor pkgs;
        in
        {
          default = pkgs.mkShell {
            # NixOS: uv's managed CPython downloads are not patchelf'd and won't run.
            UV_PYTHON_DOWNLOADS = "never";

            # Installs the git hooks (pre-commit: treefmt, gitleaks; commit-msg: committed).
            inherit (self.checks.${pkgs.stdenv.hostPlatform.system}.pre-commit) shellHook;

            packages = [
              treefmt.build.wrapper
            ]
            ++ builtins.attrValues treefmt.build.programs
            ++ (with pkgs; [
              committed
              gitleaks
              python3
              uv
              reuse
            ]);
          };
        }
      );
    };
}

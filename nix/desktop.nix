# nix/desktop.nix — GPUCLOUD Desktop (Electron) app build + wrapper
#
# `gpucloudAgent` is the fully-built `.#default` package — it ships the
# `gpucloud` binary with the venv, runtime PATH, bundled skills/plugins, etc.
# already wired up.  We point the desktop at it via the existing
# `GPUCLOUD_DESKTOP_GPUCLOUD` override env var, so the desktop's resolver
# uses our fully wrapped binary at step 4 ("existing GPUCLOUD CLI").
# No reimplementation of the agent resolution in this wrapper.
{ pkgs, lib, stdenv, makeWrapper, gpucloudNpmLib, electron, gpucloudAgent, ... }:
let
  npm = gpucloudNpmLib.mkNpmPassthru { folder = "apps/desktop"; attr = "desktop"; pname = "gpucloud-desktop"; };

  packageJson = builtins.fromJSON (builtins.readFile (npm.src + "/apps/desktop/package.json"));
  version = packageJson.version;

  # Build the renderer (dist/ + electron/ + package.json).
  renderer = pkgs.buildNpmPackage (npm // {
    pname = "gpucloud-desktop-renderer";
    inherit version;

    doCheck = false;
    # The workspace lockfile resolves all peer deps
    # correctly so --legacy-peer-deps is not needed.
    # --ignore-scripts comes from mkNpmPassthru (shared).
    makeCacheWritable = true;

    buildPhase = ''
      runHook preBuild

      # write-build-stamp.cjs replacement.  Packaged Electron reads this
      # at first-launch to pin the install.ps1 git ref; informational in
      # nix builds (the backend comes from the derivation directly).
      mkdir -p apps/desktop/build
      echo '{"schemaVersion":1,"commit":"nix","branch":"nix","dirty":false,"source":"nix"}' > apps/desktop/build/install-stamp.json

      # Build from apps/desktop/ so vite.config.ts resolves correctly.
      # The workspace root's node_modules/ is accessible as ../../node_modules/.
      cd apps/desktop

      # vite handles TS transpilation via esbuild — no type-checking.
      # We skip `tsc -b` to avoid type errors in test files that don't
      # ship in the bundle (real upstream peer-dep version mismatches
      # in @testing-library/react v16 — not blocking the build).
      # Call vite directly from root node_modules to avoid npx resolving
      # through unpatched workspace symlinks.
      node ../../node_modules/vite/bin/vite.js build --outDir dist

      # Return to source root so installPhase paths are correct.
      cd ../..

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out
      # vite writes to apps/desktop/dist/ (we cd'd there in buildPhase).
      # apps/desktop/build was created before the cd.  electron/ is source.
      cp -r apps/desktop/dist $out/
      cp -r apps/desktop/electron $out/
      cp -r apps/desktop/build $out/
      cp apps/desktop/package.json $out/
      runHook postInstall
    '';
  });
in

# Electron wrapper: nixpkgs' electron binary pointed at the renderer dir.
stdenv.mkDerivation {
  pname = "gpucloud-desktop";
  inherit version;

  dontUnpack = true;
  dontBuild = true;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/share/gpucloud-desktop $out/bin
    cp -r ${renderer}/* $out/share/gpucloud-desktop/

    # Wrap the nixpkgs electron binary to launch our app.  Set
    # GPUCLOUD_DESKTOP_GPUCLOUD to the absolute path of the nix-built `gpucloud`
    # binary so the desktop's resolver step 4 ("existing GPUCLOUD CLI on
    # PATH") uses our fully wrapped binary — venv with all deps,
    # bundled skills/plugins, runtime PATH (ripgrep/git/ffmpeg/etc).
    # No reimplementation of the agent resolver in the wrapper.
    makeWrapper ${lib.getExe electron} $out/bin/gpucloud-desktop \
      --add-flags "$out/share/gpucloud-desktop" \
      --set GPUCLOUD_DESKTOP_GPUCLOUD "${lib.getExe gpucloudAgent}" \
      --set ELECTRON_IS_DEV 0

    runHook postInstall
  '';

  passthru = {
    inherit (renderer.passthru) packageJsonPath;
  };

  meta = with lib; {
    description = "Native Electron desktop shell for GPUCLOUD Agent";
    homepage = "https://github.com/NousResearch/gpucloud-agent";
    license = licenses.mit;
    platforms = platforms.unix;
    mainProgram = "gpucloud-desktop";
  };
}

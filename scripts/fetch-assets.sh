#!/usr/bin/env bash
# Download the 17 AI-generated photos to assets/ and rewrite index.html to use
# local paths instead of the external CDN URLs. Run from the repo root on any
# machine with curl, e.g.
#
#   bash scripts/fetch-assets.sh
#
# Idempotent: re-running just refreshes the files and the rewrite is a no-op
# once the CDN URLs have been replaced.

set -euo pipefail

CDN="https://d8j0ntlcm91z4.cloudfront.net/user_33IJIDZ0cOmwdzXkCP5nbQryjVC"

# filename → CDN basename (timestamped UUID minted by the generation run)
declare -a MAP=(
  "hero.png        hf_20260513_155810_eac2b9f5-fad3-4e18-999c-6f270efcc3c4.png"
  "about-1.png     hf_20260513_155812_6478df1f-5ed9-468e-b2e6-a86ecfce290d.png"
  "about-2.png     hf_20260513_155815_872a85fa-a0f6-4321-85da-8ec2cf0b70c0.png"
  "dish-1.png      hf_20260513_155818_6aff3a1f-50c5-4565-898b-0b8c65985eec.png"
  "dish-2.png      hf_20260513_155820_5f4465a2-63fe-42b5-8437-16f601a4d835.png"
  "dish-3.png      hf_20260513_155823_914febbb-db69-48bd-a415-68aed5be475c.png"
  "dish-4.png      hf_20260513_155826_154d403e-82d0-4a73-bf72-a68b1aa6c3e7.png"
  "reserve.png     hf_20260513_155829_bd0d4937-6914-4c32-b790-54a0d7d6fc34.png"
  "g1.png          hf_20260513_155839_35bb04c6-9069-49e9-84f7-4b1db9465089.png"
  "g2.png          hf_20260513_155841_11afa6c6-3663-48fe-bcc8-288e94389143.png"
  "g3.png          hf_20260513_155844_8a6a529d-d063-49b9-8832-1929128de2bc.png"
  "g4.png          hf_20260513_155847_10a3db57-5b4f-4736-940b-7f6dd1dd1bd8.png"
  "g5.png          hf_20260513_155849_792d2ef5-692c-4417-bfe8-b4ddf5a086e1.png"
  "g6.png          hf_20260513_155852_c9f3531e-8374-4daa-be7e-c603b2bf6c75.png"
  "g7.png          hf_20260513_155855_0ae057f6-7316-4501-861b-ef959096d4ac.png"
  "g8.png          hf_20260513_155858_42b21faf-2113-4480-b691-8fe6550cffcc.png"
  "g9.png          hf_20260513_155901_14cbba0c-15a6-4436-8b1b-5f941c795043.png"
)

mkdir -p assets

echo "Downloading 17 photos -> assets/"
for row in "${MAP[@]}"; do
  read -r out src <<<"$row"
  echo "  $out  <-  $src"
  curl -fsSL -o "assets/$out" "$CDN/$src" &
done
wait
echo "Done."

# Rewrite index.html: swap every "$CDN/hf_..._<UUID>.png" for "assets/<out>.png".
echo "Rewriting index.html to reference local paths..."
python3 - <<'PY'
import re, pathlib
mapping = {
  "hf_20260513_155810_eac2b9f5-fad3-4e18-999c-6f270efcc3c4.png": "hero.png",
  "hf_20260513_155812_6478df1f-5ed9-468e-b2e6-a86ecfce290d.png": "about-1.png",
  "hf_20260513_155815_872a85fa-a0f6-4321-85da-8ec2cf0b70c0.png": "about-2.png",
  "hf_20260513_155818_6aff3a1f-50c5-4565-898b-0b8c65985eec.png": "dish-1.png",
  "hf_20260513_155820_5f4465a2-63fe-42b5-8437-16f601a4d835.png": "dish-2.png",
  "hf_20260513_155823_914febbb-db69-48bd-a415-68aed5be475c.png": "dish-3.png",
  "hf_20260513_155826_154d403e-82d0-4a73-bf72-a68b1aa6c3e7.png": "dish-4.png",
  "hf_20260513_155829_bd0d4937-6914-4c32-b790-54a0d7d6fc34.png": "reserve.png",
  "hf_20260513_155839_35bb04c6-9069-49e9-84f7-4b1db9465089.png": "g1.png",
  "hf_20260513_155841_11afa6c6-3663-48fe-bcc8-288e94389143.png": "g2.png",
  "hf_20260513_155844_8a6a529d-d063-49b9-8832-1929128de2bc.png": "g3.png",
  "hf_20260513_155847_10a3db57-5b4f-4736-940b-7f6dd1dd1bd8.png": "g4.png",
  "hf_20260513_155849_792d2ef5-692c-4417-bfe8-b4ddf5a086e1.png": "g5.png",
  "hf_20260513_155852_c9f3531e-8374-4daa-be7e-c603b2bf6c75.png": "g6.png",
  "hf_20260513_155855_0ae057f6-7316-4501-861b-ef959096d4ac.png": "g7.png",
  "hf_20260513_155858_42b21faf-2113-4480-b691-8fe6550cffcc.png": "g8.png",
  "hf_20260513_155901_14cbba0c-15a6-4436-8b1b-5f941c795043.png": "g9.png",
}
cdn_prefix = "https://d8j0ntlcm91z4.cloudfront.net/user_33IJIDZ0cOmwdzXkCP5nbQryjVC/"
src = pathlib.Path("index.html").read_text(encoding="utf-8")
hits = 0
for cdn_name, local_name in mapping.items():
  url = cdn_prefix + cdn_name
  if url in src:
    src = src.replace(url, f"assets/{local_name}")
    hits += 1
pathlib.Path("index.html").write_text(src, encoding="utf-8")
print(f"  rewrote {hits}/17 URLs")
PY

echo
echo "Done. Review with:  git diff index.html assets/"
echo "Then commit:        git add assets index.html && git commit -m 'Vendor AI photos into assets/'"

# Third-party notices

<!-- stamp -->

crabcrop is released under the MIT License, in [LICENSE](LICENSE). The
`crabcrop` binary in the release tarballs, the container images and the
Homebrew install also contains code by other authors, under their own
licenses. This file collects their notices:

1. [Native libraries](#native-libraries): the C and assembly code the binary
   links, which crates compile from sources they vendor, and dav1d.
2. [The AI image](#the-ai-image): ONNX Runtime, which only the `crabcrop-ai`
   image carries.
3. [Rust crates](#rust-crates): every crate the release build compiles, with
   the license each is used under.
4. [License texts](#license-texts): the license texts those crates ship, with
   their copyright notices.

This software is based in part on the work of the Independent JPEG Group.

## Native libraries

These libraries are compiled into the binary from C and assembly sources. Their
license texts below are copied from those sources.

### libjpeg-turbo

turbojpeg-sys compiles libjpeg-turbo from the sources it vendors and links it
statically: crabcrop writes JPEG output with it, and decodes JPEG sources at
reduced scale. libjpeg-turbo is covered by the IJG License, the Modified
(3-clause) BSD License, and, for its SIMD code, the zlib License.

<!-- include: crate=turbojpeg-sys path=libjpeg-turbo/LICENSE.md sha256=96f5b328adbb78eeaaec6980d73fd558cb1e4d62560ed615646bc3cf5e532430 -->

The IJG License, from the LEGAL ISSUES section of README.ijg:

<!-- include: crate=turbojpeg-sys path=libjpeg-turbo/README.ijg from="LEGAL ISSUES\n====" to="assumed by the product vendor." sha256=3a72a4e3fd0f2a7539b28ae589cf9f52826931b4340b2db6a06f2533f3e19a65 -->

The zlib License, from the SIMD code's common header (comment markers removed):

<!-- include: crate=turbojpeg-sys path=libjpeg-turbo/simd/nasm/jsimdext.inc from="; Copyright 2009" to="from any source distribution." uncomment=";" sha256=54d108e0bbd05f95dc468aa555aa5a7edf5d3acbcb92b193ecd309fa9284f94d -->

### libwebp

libwebp-sys compiles libwebp from the sources it vendors and links it
statically: crabcrop decodes WebP sources and writes lossy WebP output with it.
libwebp is under the BSD 3-Clause License, with Google's additional patent
grant.

<!-- include: crate=libwebp-sys path=vendor/COPYING sha256=5aec868f669e384a22372a4e8a1a6cd7d44c64cd451f960ca69cc170d1e13acf -->

<!-- include: crate=libwebp-sys path=vendor/PATENTS sha256=cc3273e0694ea5896145e0677699b53471b03ea43021ddc50e7923fbb9f5023c -->

### AWS-LC

aws-lc-sys compiles AWS-LC, the cryptography library under the HTTPS client
(rustls, through aws-lc-rs), from the sources it vendors. Its LICENSE file
names the license of each of its parts and reproduces their texts:

<!-- include: crate=aws-lc-sys path=aws-lc/LICENSE sha256=977c541bd25ffe36975dde25c028c0cd15ddf3d08243983b58e8a6a3d9447523 -->

### dav1d

The release binaries decode AVIF with dav1d 1.5.1, which
`scripts/build_dav1d.sh` builds from VideoLAN's release tarball and links
statically. (The image built from the repository's `Dockerfile` links Debian's
`libdav1d7` package instead; its copyright file is
`/usr/share/doc/libdav1d7/copyright` in that image.) dav1d is under the BSD
2-Clause License. Its x86 assembly is built with x264's x86inc.asm, under the
ISC License. As an AV1 implementation, it also carries the Alliance for Open
Media Patent License 1.0.

<!-- pinned: file=scripts/build_dav1d.sh sha256=401813f1f89fa8fd4295805aa5284d9aed9bc7fc1fdbe554af4292f64cbabe21 -->

<!-- include: file=scripts/notices/dav1d/COPYING label="`COPYING` in dav1d 1.5.1" sha256=b327887de263238deaa80c34cdd2ff3e0ba1d35db585ce14a37ce3e74ee389e9 -->

<!-- include: file=scripts/notices/dav1d/x86inc-notice.txt label="`src/ext/x86/x86inc.asm` in dav1d 1.5.1 (comment markers removed)" from=";* Copyright" to="PERFORMANCE OF THIS SOFTWARE." uncomment=";*" sha256=f82269eda688add335ea925c94b6524d0af6d4ca1179bc24716242e0c9f97c9b -->

<!-- include: file=scripts/notices/dav1d/PATENTS label="`doc/PATENTS` in dav1d 1.5.1" sha256=335eca574598bf4ca181b12f708d6669e5a5e78c8e1513e5b35fa1f03901484b -->

### AV1 assembly in rav1e and av-scenechange

rav1e, the AV1 encoder behind AVIF output, and av-scenechange, its scene
detection, carry x86 and Arm assembly, much of it from dav1d, under the BSD
2-Clause License (dav1d's text above, or rav1e's in the license texts below).
Their x86 assembly is built with x86inc.asm under the ISC License, and the
Alliance for Open Media Patent License 1.0 above covers rav1e as it covers
dav1d (rav1e's copy is its PATENTS file). The copyright notices in that
assembly:

<!-- copyrights: crate=rav1e glob=src/**/*.asm glob=src/**/*.S -->

<!-- copyrights: crate=av-scenechange glob=src/**/*.asm glob=src/**/*.S -->

<!-- include: crate=rav1e path=src/ext/x86/x86inc.asm from=";* Copyright" to="PERFORMANCE OF THIS SOFTWARE." uncomment=";*" sha256=48dd6d1b29d5b1f856a0b0d1b16b8fe381a07e1328a7124629db13c692dee33f -->

## The AI image

The `crabcrop-ai` image adds ONNX Runtime 1.31.0, which crabcrop loads at run
time from `/opt/onnxruntime/libonnxruntime.so`. ONNX Runtime is under the MIT
License, below. Its own third-party notices ship in the image beside this file,
as `/usr/share/doc/crabcrop/onnxruntime/ThirdPartyNotices.txt`, with its
`LICENSE`: `Dockerfile.ai` extracts both from the NuGet package it verifies,
the one it takes the library from.

<!-- pinned: file=Dockerfile.ai sha256=7e35fdb4f5adb62f01e60798e2e5ed05108f5fbbe1cca03fbcc37903cf7c5e3b -->

<!-- include: file=scripts/notices/onnxruntime/LICENSE label="`LICENSE` in ONNX Runtime 1.31.0" sha256=2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c -->

The image also carries two models, which `Dockerfile.ai` downloads and checks
against the SHA-256 it pins for each. Their license texts ship beside this
file, in `/usr/share/doc/crabcrop/models/`, copied byte for byte from the
projects the models come from, at the commits named below.

`/models/u2netp.onnx` (SHA-256
`309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`), the model
behind `removebg`, is u2netp, the small U²-Net model, in the ONNX form that
rembg publishes in its `v0.0.0` release. Neither project says which license
covers this file. U²-Net's repository is under the Apache License 2.0, and its
README links the trained u2netp weights from Google Drive and Baidu Pan,
outside the repository, without naming a license for them. rembg is under the
MIT License, and its README, in its list of models, says that model weights
carry their own licenses, independent of rembg's MIT license, and links
U²-Net's repository as u2netp's source. This file does not decide between the
two, and the image ships both texts: the Apache License 2.0 as
`/usr/share/doc/crabcrop/models/u2netp/U-2-Net/LICENSE` and the MIT License as
`/usr/share/doc/crabcrop/models/u2netp/rembg/LICENSE.txt`. The Apache License
2.0 asks a redistribution to give its recipients a copy of the license, to
mark the files it changes, to keep the work's copyright, patent, trademark and
attribution notices in any source form it distributes, and to carry the
attribution notices of the work's NOTICE file. U²-Net's repository has no
NOTICE file and its LICENSE names no copyright holder, and the image ships the
model unchanged. The MIT License asks that the copyright notice and the license
text go with every copy.

<!-- pinned: file=Dockerfile.ai sha256=309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8 -->

<!-- include: file=scripts/notices/models/u2netp/U-2-Net/LICENSE label="`LICENSE` in U²-Net at commit ac7e1c8, https://raw.githubusercontent.com/xuebinqin/U-2-Net/ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29/LICENSE" sha256=c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 -->

<!-- include: file=scripts/notices/models/u2netp/rembg/LICENSE.txt label="`LICENSE.txt` in rembg at commit 991927b, https://raw.githubusercontent.com/danielgatis/rembg/991927b77667946dab59c50df31f7d4640e80457/LICENSE.txt" sha256=90a3215072968fd304669c5389f04f1274a587abdd0507d99dead0f5511f8999 -->

`/models/yunet.onnx` (SHA-256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`), the face
detection model, is YuNet, as `face_detection_yunet_2023mar.onnx` in OpenCV
Zoo's `models/face_detection_yunet` directory. At the commit named below, that
file is the one `Dockerfile.ai` pins, and the directory's README says that all
files in it are licensed under the MIT License, in its `LICENSE` file, which
the image ships as `/usr/share/doc/crabcrop/models/yunet/LICENSE`. OpenCV Zoo
as a whole is under the Apache License 2.0, and its README refers to each
model's own license. The MIT License asks that the copyright notice and the
license text go with every copy.

<!-- pinned: file=Dockerfile.ai sha256=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 -->

<!-- include: file=scripts/notices/models/yunet/LICENSE label="`models/face_detection_yunet/LICENSE` in OpenCV Zoo at commit 48b10fa, https://raw.githubusercontent.com/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/LICENSE" sha256=c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958 -->

## Rust crates

<!-- crates -->

<!-- source-offers -->

<!-- notice-files -->

## License texts

<!-- licenses -->

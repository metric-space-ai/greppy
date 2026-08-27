# Greppy WinFsp transport fork

This directory defines the minimal Windows kernel/user transport fork required
by Greppy's portable CoW workspace provider. Upstream WinFsp 2.1 rejects
FileLinkInformation in the kernel before the existing FUSE link callback can
run. Greppy therefore cannot satisfy its hardlink contract by linking the
unmodified WinFsp runtime.

The fork is intentionally maintained as a patch over the immutable upstream
source commit recorded in upstream.json. It changes only the hardlink
transport:

- advertise FILE_SUPPORTS_HARD_LINKS only when the provider supplies a link
  operation;
- post FileLinkInformation through the existing SetInformation transaction;
- reserve one ABI-stable FSP_FILE_SYSTEM_INTERFACE slot for HardLink;
- route that callback to FUSE link;
- return the provider's inode link count in FILE_STANDARD_INFORMATION.

This patch is not a release artifact by itself. A release requires a
reproducible Windows build, the full WinFsp test suite including the previously
excluded link-information coverage, a signed driver loaded on a clean machine,
Greppy's real mount contract, and the exact-SHA 300k-file performance gate.

The modified WinFsp component remains GPL-3.0. Release packaging must include
the complete corresponding source, this patch, the upstream and patched source
hashes, the WinFsp copyright notice, SBOM entries, signatures and provenance.
Greppy remains a separate Apache-2.0 FLOSS work linked through WinFsp's FLOSS
exception; the exception for distributing an unmodified upstream installer
does not remove the GPL obligations for this modified driver.

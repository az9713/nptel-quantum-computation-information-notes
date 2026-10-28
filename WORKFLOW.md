# Lecture-note production and verification

The official NPTEL syllabus was matched to the 60 playlist video IDs, restoring the 12-week teaching order while preserving original upload positions.

1. Acquire timestamped source text: 38 English auto-caption tracks and 22 local faster-whisper small.en transcripts. Keep media, full transcripts, downloaded papers and working frames private.
2. Read assigned text sequentially; inspect selected frames and primary references for equations or scientific corrections. Distinguish source coverage from a full audiovisual review.
3. Develop each approved chapter directly in HTML, defining symbols and assumptions before use and separating physical postulates, local proofs, cited results and numerical examples.
4. Run the complete laboratory and verify every worked solution. Embed the exact runnable source and saved stdout.
5. Obtain two independent agent editorial review rounds per chapter, apply required repairs and confirm the current canonical body digest. 40 review reports support 20 accepted final chapters.
6. Assemble the offline reader with local MathJax and extension assets. Check all 20 chapters at desktop and mobile sizes: 3,471 expressions per full edition rendered with zero mathematics errors. Check controls, search and overflow.
7. Validate every local link, source-duration bound, full code/output embed and execution of all 20 labs. Verify a copied export without the private authoring inputs.

The evidence files describe distinct checks. Editorial acceptance is not formal proof verification. Finite program checks are not experimental validation. Selected-frame inspection is not comprehensive audiovisual review.

8. Publish only the portable reader, sidecars, companion code and verification tools on the public main branch. Serve the root through GitHub Pages at https://az9713.github.io/nptel-quantum-computation-information-notes/. Record served-file integrity and live reader checks separately in publication-validation.json.

The public tools/build_book.py refreshes navigation without private sources or authoring files and refuses changed chapter bodies until they are reviewed and their metadata is updated.

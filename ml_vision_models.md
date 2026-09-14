# ILI ML and vision models

Catalog of the models currently in use. For each family: industry use case, architecture, and where that design tends to fit.

Several families share a backbone. Those patterns are summarized first; the named models below are the trained specializations.

---

## Shared architecture patterns

**DINOv2-LSTM.** A small Vision Transformer (DINOv2 ViT-S/14) encodes each ultrasound frame. An LSTM reads a short temporal window of those embeddings so nearby frames can vote on the same event. A linear head then scores one anomaly type per frame. Sliding windows with triangular weighting reduce edge artifacts. This is the usual choice when the signal is a *sequence of images* and the event lasts several frames.

**ResNet classifier on arm-angle images.** A compact CNN (ResNet18) on a 2-D plot of mechanical arm angle vs. distance. Cheap to run, good for a coarse “is something here?” score, weaker at precise localization.

**Two-stage object detectors (Faster R-CNN, RF-DETR, Deformable-DETR).** A backbone extracts a feature map; a detection head proposes boxes and classes. R-CNN variants use region proposals and an FPN; DETR variants treat detection as set prediction with transformers. Natural fit for *localized objects* in a rendered view (tubeview, channel image).

**Encoder–decoder segmentation (UNet, DarkFormer / SAM2).** A hierarchical encoder (ResNet or SAM2 Hiera) produces multi-scale features; a decoder upsamples them into a dense mask. Used when the answer is a *surface or region*, not a single score or box.

**Student–teacher distillation.** A frozen teacher and a smaller student both see the same image. Distance between their tokens is the anomaly score. Useful when labeled defects are rare and “normal” data is plentiful.

---

## Named models

### ID corrosion (`IDCorrosionDetection`)

**Industry use.** Internal metal loss on the pipe bore: pitting, grooving, and general corrosion from product, water, or CO₂/H₂S. This is a primary integrity threat for liquid and gas transmission lines; operators need length, location, and confidence so they can prioritize digs.

**Architecture.** DINOv2-S/14 + LSTM + corrosion head. Input is a sequence of beamformed ultrasound frames looking at the inner wall.

**Where it is useful.** Any sequential NDT view of an inner surface (ultrasound, similar B-scans) where corrosion appears as a multi-frame brightness or geometry change rather than a single snapshot.

### OD corrosion (`ODCorrosionDetection`)

**Industry use.** External metal loss under coating, soil, or insulation (for example coating holidays, MIC, or disbonded coating corrosion). OD features are often weaker and more spread out than ID pits, so they are treated as a separate detection problem.

**Architecture.** Same DINOv2-LSTM layout, trained on long-long (L-L) images. Later versions pool over patch tokens so the head can attend to a region instead of a single token.

**Where it is useful.** Outer-wall or through-wall ultrasound modes, and more generally any clip-level detector that must find faint, spatially extended defects on a second imaging geometry.

### Dent, ultrasound (`DentDetection`)

**Industry use.** Mechanical damage from rocks, construction, or third-party contact. Dents concentrate stress and can interact with corrosion or welds; codes treat them as reportable anomalies.

**Architecture.** DINOv2-LSTM with a dent head on the same ultrasound clip stream as ID corrosion. Scores dent probability along the frame sequence.

**Where it is useful.** When a dent changes the ultrasound image over several frames (standoff, wall geometry, or reverberation) and you already run a clip transformer. Complements a mechanical (arm-angle) dent detector rather than replacing it.

### Lamination (`LaminationDetection`)

**Industry use.** Mid-wall planar flaws from steel manufacture (inclusions, rolling laminations) or hydrogen-induced cracking. They can hide wall-loss estimates and are a concern at welds and in sour service.

**Architecture.** DINOv2-LSTM with a lamination head. Later weights freeze most of the backbone and attach LoRA or DoRA adapters so a new task can be added without retraining the full network.

**Where it is useful.** Mid-wall / interface reflectors in ultrasound sequences, and as a pattern for adding a new ILI task onto a shared backbone with parameter-efficient fine-tuning.

### Seam weld (`SeamWeldDetection`)

**Industry use.** Locating the longitudinal or helical mill seam. Seam position is needed to associate anomalies with the weld (ERW/SAW issues, hook cracks, lack of fusion) and to report o’clock orientation.

**Architecture.** DINOv2-LSTM with a seam-weld head. Frame-level presence score along the clip.

**Where it is useful.** Continuous, slowly varying landmarks in sequential imagery—seams, beads, or similar linear features that should be tracked, not boxed.

### Dent, arm encoder (`DentDetectionAE`)

**Industry use.** Same integrity concern as ultrasound dents, but from the tool’s mechanical arms rather than the acoustic image. Arm deflection is a direct geometric cue that a diameter restriction or dent is present.

**Architecture.** ResNet18 classifier on short axial windows of arm-angle images. Multi-label head: one probability per track.

**Where it is useful.** Fast, whole-ring screening from caliper / arm data; onboard or first-pass filters; any setting where a small CNN on a 2-D “angle vs. distance” image is enough and a transformer on ultrasound is not required.

### Channel crack (`ChannelCrackDetection`)

**Industry use.** Axial or circumferential cracking (SCC, fatigue, hook cracks, seam-weld cracks). Cracks are among the highest-consequence ILI calls; they need a box, not just a clip score, so analysts can review the channel image.

**Architecture.** Object detector. Older weights: Faster R-CNN, ResNet-18 backbone, first without then with a feature pyramid. Current weights: Deformable-DETR (transformer detector with multi-scale deformable attention). Output is bounding boxes and confidence.

**Where it is useful.** Localized, thin defects in 2-D inspection images. R-CNN is a strong baseline; Deformable-DETR is a better fit when objects vary a lot in scale and you want end-to-end set prediction without hand-tuned anchors.

### Girth weld, arm (`ArmGirthWeldDetection`)

**Industry use.** Finding girth welds (the joints between pipe joints). Weld locations are the backbone of the inspection: they define joints, support distance correlation to the pipe tally, and are themselves inspection features (misalignment, undercut, cracks).

**Architecture.** ResNet18 on arm-angle patches; also shipped as ONNX (fp32/fp16). Outputs a coarse probability that a weld sits in that view-distance window.

**Where it is useful.** Cheap first-stage weld finding from geometry, edge devices, or long inspections where you only need “weld-like bump in this meter” before a slower image model refines it.

### Girth weld, tubeview (`TubeviewGirthWeldDetection`)

**Industry use.** Precise girth-weld stationing after a coarse candidate is known. Accurate weld position is what lets you match the run to the as-built tally and report features relative to joints.

**Architecture.** UNet with a ResNet34 encoder on four-channel 224×224 tubeviews (full, outer, inner, distance-to-max). Dense segmentation mask, then a peak/gap rule to pick the weld station.

**Where it is useful.** Refining a known candidate in a rendered unwrapped pipe view. The same encoder–decoder pattern applies to other “find the bright/dark band” problems in tubeviews or B-scans.

### Sleeve weld (`SleeveWeldDetection`)

**Industry use.** Repair sleeves (full-encirclement or similar) and the fillet welds at their ends. Sleeves change remaining-life calculations and must not be confused with girth welds or heavy wall.

**Architecture.** RF-DETR Nano (transformer object detector) on overlapping multitrack tubeview patches. Single class; a follow-up brightness check on a tight crop rejects lookalikes.

**Where it is useful.** Sparse, box-shaped objects on a long panoramic image. RF-DETR is a modern, relatively small detector for that setting; a simple photometric gate is typical when the class is rare and false positives are costly.

### Pipeline components (`TubeviewComponentDetection`)

**Industry use.** Appurtenances along the line: fittings, valves, stopples, and similar hardware. These are landmarks for mapping, excavation planning, and for excluding non-pipe features from metal-loss calls.

**Architecture.** Faster R-CNN with a ResNeXt-101-32x8d feature pyramid. Standard two-stage detector: region proposals, then box + class. Current training uses three foreground classes (fitting, valve, stopple).

**Where it is useful.** Multi-class objects that are large and distinctive in a tubeview. A heavy FPN backbone is appropriate when class confusion (valve vs. fitting) matters more than latency.

### ID surface (`IDSurfaceDetection`)

**Industry use.** Inner-wall surface pick: the time/depth of the ID echo in each beamformed frame. That surface is the reference for wall thickness, corrosion sizing, and OD imaging priors.

**Architecture.** DarkFormer v5: SAM2 Hiera-S image encoder, SAM memory/prompt modules removed, U-Net-style decoder with receptive-field blocks and boundary-aware attention. Input is a 256×256 RGB-replicated B-scan; output is a raster surface, then vectorized.

**Where it is useful.** Dense “draw the interface” problems in noisy ultrasound, analogous to medical or industrial surface/segmentation tasks where a strong pretrained hierarchical encoder plus a light decoder works better than a from-scratch UNet.

### OD surface (`ODSurfaceDetection`)

**Industry use.** Outer-wall surface pick, used with the ID surface to get remaining wall and to size external metal loss.

**Architecture.** DarkFormer v4: same Hiera-S trunk, adapters on the frozen encoder, dedicated OD decoder. Typically given the ID surface as a geometric prior.

**Where it is useful.** A second, weaker interface that should stay consistent with an already-estimated inner surface—stacked-echo or dual-surface tracking in NDT.

### Anomaly distillation (`dark_ad`)

**Industry use.** Catch-all / unknown defect screening: flag frames that do not look like the training “normal” population (unusual geometry, rare damage, sensor artifacts). Complements supervised heads that only fire on named classes.

**Architecture.** Student–teacher DINOv2 ViT-B/14. Teacher frozen; student distilled on in-distribution frames. Anomaly score is cosine distance on class and patch tokens, with Gaussian smoothing and running z-scores.

**Where it is useful.** Inspection or manufacturing lines where most data is healthy, labels for every failure mode do not exist, and you want a heatmap of “this does not match normal” rather than a named class.

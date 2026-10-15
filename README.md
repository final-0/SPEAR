# SPEAR: Staged Pre-training and Expert Adapter Routing for Generalizable Optical Flow Learning

Code for our optical-flow entry to the [RoCo Spring challenge](https://roco-spring.github.io/index.html)
(Large encoder: Depth Anything V2 ViT-L), built on [WAFT](https://github.com/princeton-vl/WAFT).

1. **Clean training** builds a strong 2-frame model (the *parent*).
2. **Robust training** freezes the parent and adds a router and corruption-specific adapters,
   which stay switched off on clean input.

Project page: <https://final-0.github.io/SPEAR/> · Method details: [docs/spear_method.md](docs/spear_method.md)

## Pipeline

Every step is one Slurm job in `jobs/`, run from the repository root in this order:

```
prepare-step1-venv               Python venv inside the container
prepare-step2-spring-val         split Spring train into train + val (run once)
prepare-step3-weather-overlays   frost / rain / snow layers, extracted from RobustSpring test images

clean-step1-tartanair            TartanAir            (from the WAFT checkpoint)
clean-step2-tartanair            TartanAir
clean-step3-flyingthings         FlyingThings3D
clean-step4-spring               Spring               -> parent model

robust-step1-router              router               -> step-010000.pt
robust-step2-adapters            adapters             -> step-005000.pt (final model)

clean-eval  / clean-submit  / clean-package     parent (2-frame): Spring
robust-eval / robust-submit / robust-package    final model (3-frame): RoCo
```

## Setup

**1. Data and weights** (download from the original sources):

| Item | Source | Put it at |
|---|---|---|
| Spring: `train_frame_{left,right}`, `train_flow_{FW,BW}_{left,right}`, `test_frame_{left,right}` | [DOI 10.18419/DARUS-3376](https://doi.org/10.18419/DARUS-3376) | `$SPRING_ROOT` |
| RobustSpring: all 20 corruption archives | [DOI 10.18419/DARUS-5047](https://doi.org/10.18419/DARUS-5047) | `$ROBUST_SPRING_ROOT` |
| FlyingThings3D: `frames_cleanpass`, `optical_flow` | [Scene Flow datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) | `$THINGS_ROOT` |
| TartanAir: `image_left`, `flow` (Easy and Hard) | [TartanAir](https://theairlab.org/tartanair-dataset/) | `$TARTAN_ROOT` |
| Depth Anything V2 ViT-L: `depth_anything_v2_vitl.pth` | [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | `depth-anything-ckpts/` |
| WAFT checkpoint: `dav2/spring.pth` | [WAFT Model Zoo](https://github.com/princeton-vl/WAFT#model-zoo) | `ckpts/dav2/spring.pth` |
| Spring subsampling tools | [download](https://cloud.visus.uni-stuttgart.de/index.php/s/J33vJ3meZf7Jeq1) | `$SUBSAMPLING_TOOLS` |

SHA-256: `spring.pth` `e57055120aab42dd1fe91cb7cc6ac9166df968a167cc089966e3f5845b0e6bea`,
`depth_anything_v2_vitl.pth` `a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345`.

Nothing else needs to be downloaded: the Spring val split and the weather overlays are generated
by the `prepare-*` jobs.

**2. Paths** — every machine-specific setting lives in one file:

```bash
cp spear_env.sh.example spear_env.sh   # edit it (each variable is commented)
source spear_env.sh                    # in every shell you submit jobs from
```

**3. Container and venv**

```bash
docker build -t spear:pt25 .
singularity build "$SPEAR_CONTAINER" docker-daemon://spear:pt25   # or apptainer build
sbatch jobs/prepare-step1-venv.sh
```

The `#SBATCH` lines assume single nodes with 1, 4 or 8 GPUs (A100 40 GB in our runs); adjust them
to your cluster.

## Run

```bash
sbatch jobs/prepare-step2-spring-val.sh
sbatch jobs/prepare-step3-weather-overlays.sh

sbatch jobs/clean-step1-tartanair.sh
sbatch jobs/clean-step2-tartanair.sh
sbatch jobs/clean-step3-flyingthings.sh
sbatch jobs/clean-step4-spring.sh

sbatch jobs/robust-step1-router.sh
sbatch jobs/robust-step2-adapters.sh
```

Each job needs the output of the previous one; chain them with `sbatch --dependency=afterok:<jobid>`.
Clean checkpoints go to `checkpoints/`, everything else to `$SPEAR_OUTPUT`, logs to `logs/`.
Optional per-job settings (e.g. `CKPT`) are listed at the top of each job script.

## Evaluate and submit

```bash
sbatch jobs/clean-eval.sh                                     # parent, Spring train/val EPE
sbatch jobs/robust-eval.sh                                    # parent vs final model, Spring val + corruptions

sbatch jobs/robust-submit.sh && bash jobs/robust-package.sh   # RoCo: clean + 20 corruptions
sbatch jobs/clean-submit.sh  && bash jobs/clean-package.sh    # Spring only, 2-frame parent
```

The `*-package.sh` scripts run on CPU for several hours (use tmux or similar) and write the
`.hdf5` files to `<output>/_uploads/`. The RoCo `.flo5` outputs take about 1.1 TB; delete them
after packaging.

## Notes

- `prepare-step2-spring-val` **moves** files from `train/` to `val/` and refuses to run twice.
- TartanAir training uses only the trajectories listed in `config/tartanair_trajectories.json`
  (all Hard, about half of Easy). Extra trajectories in `$TARTAN_ROOT` are ignored.
- Robust training checks the parent's SHA-256. If you trained your own parent, set
  `SPEAR_PARENT` and `SPEAR_PARENT_SHA256` in `spear_env.sh` before the robust jobs.
- Robust training runs need a new, empty output directory.
- The weather overlays, corruption severities and the impulse-noise filter threshold were derived
  from RobustSpring test images. No test ground truth is used.

## License

Code derived from [WAFT](https://github.com/princeton-vl/WAFT) (BSD 3-Clause, see `LICENSE`).
`thirdparty/DepthAnythingV2` is from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2);
see that repository for the licenses of the code and weights. Datasets are subject to their own terms.

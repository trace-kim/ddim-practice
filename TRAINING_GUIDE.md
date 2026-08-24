# DDIM Training Quick Start / DDIM 학습 빠른 시작

[English](#english) | [한국어](#korean)

---

<a id="english"></a>

## English

This guide covers the shortest supported path for starting training and watching its progress. Run all commands from the repository root.

### 1. Install

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On the workstation that will run the MLflow results UI, also install:

```text
python -m pip install -e ".[tracking]"
```

For an offline HPC, use the approved environment or wheelhouse described in the [full workflow guide](docs/training_workflow.md).

### 2. Configure and check the machine

Create a machine profile once. Choose a short ID and reuse it in later commands:

```text
ddimctl machine configure --id local-4060ti
```

Use dataset alias `sem`. Select the executor that matches the machine:

- Windows: `windows-task`
- HPC with `sbatch`: `slurm`
- HPC reached through a corporate job portal: `external-hpc`

Enter paths that exist on that machine, then check the setup:

```text
ddimctl doctor --machine local-4060ti --exercise-executor
```

Do not start a long run until the required checks pass.

### 3. Start training

Use the wizard so you do not need to type a long training command:

```text
ddimctl train wizard --machine local-4060ti
```

During the wizard:

1. Enter a clear label such as `sem32-baseline`.
2. Press Enter to keep a value from `configs/sem.yml`, or type a new value.
3. Review the resolved settings and exact command.
4. Confirm the launch.
5. Copy the path printed after `Created run bundle:`.

With `windows-task` or `slurm`, closing the terminal does not stop training. `external-hpc` creates `worker.sh`; submit that file through the approved corporate portal. Do not use `foreground` for a long run.

### 4. Watch progress

Set the run path printed by the wizard:

```powershell
# Windows PowerShell
$run = 'E:\path\to\the\created\run'
```

```bash
# Linux
run='/path/to/the/created/run'
```

Check the current state:

```text
ddimctl run status "$run"
```

Follow the training log:

```text
ddimctl run logs "$run" --stream stdout --follow
```

Press Ctrl+C to stop following the log. Training continues.

Start the live TensorBoard UI:

```text
tensorboard --logdir "$run/tensorboard" --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006> to watch loss, validation, and generated samples.

### 5. Review completed results

TensorBoard is the live progress UI. MLflow is the detailed review UI for completed runs. This project does not include a separate custom results website.

After `ddimctl run status "$run"` reports `completed`:

```text
ddimctl track serve --port 5000
```

Wait until <http://127.0.0.1:5000> opens, then publish the run:

```text
ddimctl track publish "$run" --tracking-uri http://127.0.0.1:5000 --experiment ddim-sem
```

In MLflow, select **Model training**, open `ddim-sem`, and select the run.

For an offline HPC run, copy the entire run folder to the workstation before publishing it.

### 6. Stop or resume

Request a safe stop that saves a checkpoint:

```text
ddimctl run stop "$run"
```

After the run has stopped, resume it with the same settings:

```text
ddimctl run resume "$run"
```

Use `--force` only when a normal stop cannot work. It may lose progress after the latest checkpoint.

---

<a id="korean"></a>

## 한국어

이 문서는 학습을 시작하고 진행 상황을 확인하는 데 필요한 최소 절차만 설명합니다. 모든 명령은 저장소 최상위 폴더에서 실행합니다.

### 1. 설치

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

MLflow 결과 화면을 실행할 워크스테이션에는 다음 패키지도 설치합니다.

```text
python -m pip install -e ".[tracking]"
```

인터넷이 차단된 HPC에서는 [전체 워크플로 가이드](docs/training_workflow.md)에 설명된 승인된 환경 또는 wheelhouse를 사용합니다.

### 2. 머신 설정 및 확인

머신 프로필은 처음 한 번만 만듭니다. 짧은 ID를 정하고 이후 명령에서도 같은 ID를 사용합니다.

```text
ddimctl machine configure --id local-4060ti
```

데이터셋 별칭은 `sem`을 사용합니다. 머신에 맞는 실행 방식을 선택합니다.

- Windows: `windows-task`
- `sbatch`를 사용할 수 있는 HPC: `slurm`
- 기업용 작업 포털을 통해 접속하는 HPC: `external-hpc`

해당 머신에서 실제로 접근할 수 있는 경로를 입력한 뒤 설정을 검사합니다.

```text
ddimctl doctor --machine local-4060ti --exercise-executor
```

필수 검사를 모두 통과하기 전에는 장시간 학습을 시작하지 마세요.

### 3. 학습 시작

긴 학습 명령을 직접 입력하지 않도록 마법사를 사용합니다.

```text
ddimctl train wizard --machine local-4060ti
```

마법사에서는 다음 순서로 진행합니다.

1. `sem32-baseline`처럼 알아보기 쉬운 실행 이름을 입력합니다.
2. `configs/sem.yml`의 값을 유지하려면 Enter를 누르고, 변경하려면 새 값을 입력합니다.
3. 최종 설정과 실제 실행 명령을 확인합니다.
4. 학습 시작 여부를 확인합니다.
5. `Created run bundle:` 뒤에 표시된 경로를 복사합니다.

`windows-task` 또는 `slurm`을 사용하면 터미널을 닫아도 학습이 계속됩니다. `external-hpc`는 `worker.sh`를 생성하므로 승인된 기업 포털을 통해 이 파일을 제출해야 합니다. 장시간 학습에는 `foreground`를 사용하지 마세요.

### 4. 진행 상황 확인

마법사가 출력한 실행 경로를 변수에 저장합니다.

```powershell
# Windows PowerShell
$run = 'E:\path\to\the\created\run'
```

```bash
# Linux
run='/path/to/the/created/run'
```

현재 상태를 확인합니다.

```text
ddimctl run status "$run"
```

학습 로그를 실시간으로 확인합니다.

```text
ddimctl run logs "$run" --stream stdout --follow
```

Ctrl+C를 누르면 로그 보기만 종료되고 학습은 계속됩니다.

실시간 TensorBoard 화면을 시작합니다.

```text
tensorboard --logdir "$run/tensorboard" --host 127.0.0.1 --port 6006
```

브라우저에서 <http://127.0.0.1:6006>을 열어 손실값, 검증 지표, 생성 샘플을 확인합니다.

### 5. 완료된 결과 확인

TensorBoard는 실시간 진행 상황을 보는 화면입니다. MLflow는 완료된 실행을 자세히 검토하는 화면입니다. 이 프로젝트에는 별도의 자체 제작 결과 웹사이트가 없습니다.

`ddimctl run status "$run"`이 `completed`를 표시하면 다음 명령을 실행합니다.

```text
ddimctl track serve --port 5000
```

브라우저에서 <http://127.0.0.1:5000>이 열릴 때까지 기다린 뒤 실행 결과를 등록합니다.

```text
ddimctl track publish "$run" --tracking-uri http://127.0.0.1:5000 --experiment ddim-sem
```

MLflow에서 **Model training** → `ddim-sem` → 실행 이름을 선택합니다.

인터넷이 차단된 HPC에서 실행한 경우, 전체 실행 폴더를 워크스테이션으로 복사한 뒤 등록합니다.

### 6. 중지 또는 재개

체크포인트를 저장하고 안전하게 중지합니다.

```text
ddimctl run stop "$run"
```

실행이 중지된 뒤 같은 설정으로 학습을 재개합니다.

```text
ddimctl run resume "$run"
```

정상 중지가 동작하지 않을 때만 `--force`를 사용하세요. 마지막 체크포인트 이후의 진행 내용이 손실될 수 있습니다.

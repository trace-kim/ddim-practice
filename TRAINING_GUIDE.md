# Training Quick Start / 학습 빠른 시작

This is the shortest supported path for starting a training run and watching it safely.

이 문서는 학습을 안전하게 시작하고 진행 상황을 확인하는 가장 간단한 방법을 설명합니다.

Run all commands from the repository root. Replace `local-4060ti` with your configured machine ID when needed.

모든 명령은 저장소 최상위 폴더에서 실행합니다. 필요하면 `local-4060ti`를 설정한 머신 ID로 바꾸세요.

## 1. Install / 설치

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

Install the optional tracking package only on a machine that will run the MLflow results UI:

MLflow 결과 UI를 실행할 머신에서만 선택 사항인 tracking 패키지를 설치합니다.

```text
python -m pip install -e ".[tracking]"
```

On an offline HPC, use the approved prebuilt environment or wheelhouse described in the [full workflow guide](docs/training_workflow.md).

인터넷이 없는 HPC에서는 [전체 워크플로 가이드](docs/training_workflow.md)에 설명된 승인된 사전 구축 환경 또는 wheelhouse를 사용하세요.

## 2. Configure the machine once / 머신 최초 설정

```text
ddimctl machine configure --id local-4060ti
```

Enter these values when prompted:

메시지가 표시되면 다음 값을 입력합니다.

- Dataset alias / 데이터셋 별칭: `sem`
- Executor: `windows-task` on Windows; `slurm` when `sbatch` is available; otherwise `external-hpc` for a corporate portal
- 실행 방식: Windows에서는 `windows-task`, `sbatch`를 사용할 수 있으면 `slurm`, 기업 포털을 사용하면 `external-hpc`
- Dataset path, runs path, Python path, and expected GPU / 데이터셋 경로, 실행 결과 저장 경로, Python 경로, 예상 GPU

Check the setup before training:

학습 전에 설정을 확인합니다.

```text
ddimctl doctor --machine local-4060ti --exercise-executor
```

Do not start a long run until every required check passes.

필수 항목이 모두 통과하기 전에는 장시간 학습을 시작하지 마세요.

## 3. Start training / 학습 시작

Use the guided wizard to avoid typing a long command:

긴 명령을 직접 입력하지 않도록 안내 마법사를 사용합니다.

```text
ddimctl train wizard --machine local-4060ti
```

1. Enter a clear run label, such as `sem32-baseline`.
2. Press Enter to accept a value from `configs/sem.yml`, or enter a new value.
3. Review the displayed settings and exact command.
4. Confirm the launch.
5. Copy the path printed after `Created run bundle:`. You will use it below.

1. `sem32-baseline`처럼 알아보기 쉬운 실행 이름을 입력합니다.
2. `configs/sem.yml`의 값을 사용하려면 Enter를 누르고, 바꾸려면 새 값을 입력합니다.
3. 표시된 설정과 실제 실행 명령을 확인합니다.
4. 학습 시작 여부를 확인합니다.
5. `Created run bundle:` 뒤에 표시되는 경로를 복사합니다. 아래 명령에서 사용합니다.

With `windows-task` or `slurm`, closing the launch terminal does not stop training. Do not use the `foreground` executor for a long run.

`windows-task` 또는 `slurm`을 사용하면 실행 터미널을 닫아도 학습이 중단되지 않습니다. 장시간 학습에는 `foreground`를 사용하지 마세요.

## 4. Watch progress / 진행 상황 확인

Set the run path printed by the wizard.

마법사가 출력한 실행 경로를 지정합니다.

Windows PowerShell:

```powershell
$run = 'E:\path\to\the\created\run'
```

Linux:

```bash
run='/path/to/the/created/run'
```

Check the current state:

현재 상태를 확인합니다.

```text
ddimctl run status "$run"
```

Follow the training log. Press Ctrl+C to stop following the log; this does not stop training.

학습 로그를 실시간으로 봅니다. Ctrl+C는 로그 보기만 종료하며 학습은 계속됩니다.

```text
ddimctl run logs "$run" --stream stdout --follow
```

Open the live TensorBoard UI:

실시간 TensorBoard UI를 엽니다.

```text
tensorboard --logdir "$run/tensorboard" --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006> in a browser. TensorBoard is for live loss, validation, and sample checks.

브라우저에서 <http://127.0.0.1:6006>을 엽니다. TensorBoard에서 실시간 손실값, 검증 지표, 생성 샘플을 확인할 수 있습니다.

## 5. Review completed results / 완료 결과 확인

TensorBoard is the live UI. MLflow is the deeper review UI for completed runs; this project does not provide a separate custom web results UI.

TensorBoard는 실시간 UI입니다. MLflow는 완료된 실행을 자세히 검토하는 UI이며, 이 프로젝트에는 별도의 자체 제작 웹 결과 UI가 없습니다.

After `ddimctl run status "$run"` reports `completed`, start the local MLflow UI:

`ddimctl run status "$run"`이 `completed`를 표시하면 로컬 MLflow UI를 시작합니다.

```text
ddimctl track serve --port 5000
```

Wait until <http://127.0.0.1:5000> opens, then publish from another terminal.

<http://127.0.0.1:5000>이 열릴 때까지 기다린 다음, 다른 터미널에서 결과를 등록합니다.

Publish the completed run:

완료된 실행 결과를 등록합니다.

```text
ddimctl track publish "$run" --tracking-uri http://127.0.0.1:5000 --experiment ddim-sem
```

Open <http://127.0.0.1:5000>, select **Model training**, open `ddim-sem`, and select your run. MLflow shows the command, parameters, metric charts, and samples.

브라우저에서 <http://127.0.0.1:5000>을 열고 **Model training** → `ddim-sem` → 실행 이름을 선택합니다. MLflow에서 명령, 설정, 지표 그래프와 생성 샘플을 확인할 수 있습니다.

For an offline HPC, copy the complete run folder to the Windows workstation first, then publish it there.

인터넷이 없는 HPC에서는 전체 실행 폴더를 Windows 워크스테이션으로 복사한 뒤 그곳에서 등록하세요.

## 6. Stop or resume / 중지 또는 재개

Request a safe stop that saves a checkpoint:

체크포인트를 저장하고 안전하게 중지합니다.

```text
ddimctl run stop "$run"
```

Resume the same run after it has stopped:

중지된 실행을 같은 설정으로 이어서 학습합니다.

```text
ddimctl run resume "$run"
```

Use `--force` only when a normal stop cannot work because it may lose progress after the latest checkpoint.

정상 중지가 동작하지 않을 때만 `--force`를 사용하세요. 마지막 체크포인트 이후의 진행 내용이 손실될 수 있습니다.

## Daily workflow / 매번 사용하는 순서

1. Activate the virtual environment. / 가상 환경을 활성화합니다.
2. Run `ddimctl doctor --machine local-4060ti`. / `ddimctl doctor --machine local-4060ti`를 실행합니다.
3. Start with `ddimctl train wizard --machine local-4060ti`. / `ddimctl train wizard --machine local-4060ti`로 시작합니다.
4. Copy the created run path. / 생성된 실행 경로를 복사합니다.
5. Watch logs and TensorBoard. / 로그와 TensorBoard를 확인합니다.
6. After completion, publish to MLflow. / 완료 후 MLflow에 등록합니다.

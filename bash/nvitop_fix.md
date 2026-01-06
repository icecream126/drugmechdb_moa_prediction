# nvitop -m 오류 해결 방법

## 문제
`nvitop -m` 실행 시 다음과 같은 오류 발생:
```
ERROR: Failed to initialize `curses` (curs_set() returned ERR)
```

## 해결 방법

### 방법 1: --once 옵션 사용 (권장)
비대화형 모드에서 한 번만 실행하고 종료:
```bash
nvitop -m --once
```

### 방법 2: TERM 환경 변수 설정
터미널 타입을 명시적으로 설정:
```bash
TERM=xterm-256color nvitop -m
```

### 방법 3: 래퍼 스크립트 사용
제공된 스크립트 사용:
```bash
bash/bash/nvitop_monitor.sh
```

### 방법 4: watch와 함께 사용
주기적으로 업데이트하려면:
```bash
watch -n 1 'nvitop --once'
```

## 참고
- `-m` 또는 `--minimal`: 최소한의 정보만 표시
- `--once`: 한 번 실행 후 종료 (비대화형 모드)
- `--force-color`: 색상 강제 사용


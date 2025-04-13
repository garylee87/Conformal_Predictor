# Ninja 사용 가능 여부 확인하기
 - Ninja는 속도가 빠른 빌드 시스템으로, Detectron2 설치 시 빌드 속도를 향상시킬 수 있습니다. Ninja가 시스템에 설치되어 있는지 확인하는 방법은 다음과 같습니다:

### Ninja 설치 확인
터미널에서 다음 명령어를 실행하여 Ninja가 설치되어 있는지 확인할 수 있습니다:

$ ninja --version

이 명령어가 버전 정보를 반환하면 Ninja가 이미 설치되어 있는 것입니다.

### Ninja 설치하기
Ninja가 설치되어 있지 않다면, 다음 방법으로 설치할 수 있습니다:

sudo apt-get update
sudo apt-get install ninja-build

or

pip install ninja


### Ubuntu/Debian 시스템의 경우:
 - pip를 통한 설치:
    - Ninja를 사용한 Detectron2 설치
    - Ninja를 사용하여 Detectron2를 설치하려면, 설치 명령어에 CMAKE_ARGS를 추가하면 됩니다:

이 방법으로 Ninja 빌드 시스템을 사용하여 Detectron2를 빌드하면 일반적으로 빌드 속도가 향상됩니다.

### OpenCV에 필요한 OpenGL 설치 (libGL.so1)
apt-get install -y libgl1-mesa-glx
# YOLOX Object Detection for Flutter

Flutter로 구현한 YOLOX 모델 기반 객체 인식 앱입니다.

## Features

- 실시간으로 사진을 촬영해서 객체 탐지
- 갤러리에서 선택한 이미지에 대한 객체 탐지
- 탐지된 객체에 대한 바운딩 박스 및 레이블 표시

## Quick Start

### Dependencies

- Flutter SDK
- Xcode

### Installation

1. 이 저장소를 클론합니다:

   ```
   git clone https://github.com/your-username/yolox-flutter.git
   ```

2. 프로젝트 디렉토리로 이동합니다:

   ```
   cd yolox-flutter
   ```

3. 종속성을 설치합니다:

   ```
   flutter pub get
   ```

4. 애플리케이션을 실행합니다:
   ```
   flutter run
   ```

## Usage

1. 앱을 실행하면 카메라 뷰가 표시됩니다.
2. 화면 하단의 버튼을 사용하여 갤러리에서 이미지를 선택하거나 카메라 모드를 전환할 수 있습니다.
3. 탐지된 객체는 색상이 지정된 바운딩 박스와 레이블로 표시됩니다.

## Project Structure

- `lib/`: Dart 소스 코드
  - `main.dart`: 앱의 진입점
  - `colors.dart`: 색상 파일
  - `yolox_detector.dart`: YOLOX 모델 추론 로직
- `assets/`: 모델 파일 및 레이블
- `android/` 및 `ios/`: 플랫폼별 코드

## Customization

- `assets/yolox.tflite`를 다른 YOLOX 모델로 교체하여 다른 버전의 YOLOX를 사용할 수 있습니다.
- `lib/yolox_detector.dart`의 `_labels` 리스트를 수정하여 탐지할 클래스를 변경할 수 있습니다.

## License

이 프로젝트는 Apache-2.0 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## Acknowledgments

- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
- TensorFlow Lite: https://www.tensorflow.org/lite
- Flutter: https://flutter.dev/

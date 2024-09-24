import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:flutter/services.dart';
import 'yolox_detector.dart';
import 'dart:ui' as ui;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark().copyWith(
        primaryColor: Colors.blueGrey[900],
        scaffoldBackgroundColor: Colors.grey[900],
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.blueGrey[900],
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blueGrey[700],
            foregroundColor: Colors.white,
          ),
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<String> uploadedImages = []; // 업로드된 이미지 경로 리스트
  final ImagePicker _picker = ImagePicker(); // 이미지 선택을 위한 ImagePicker 인스턴스
  late YOLOXDetector _detector; // YOLOX 객체 검출기
  ui.Image? _detectionResult; // 객체 검출 결과 이미지
  bool _showResult = false; // 결과 화면 표시 여부

  @override
  void initState() {
    super.initState();
    _detector = YOLOXDetector(); // YOLOX 검출기 초기화
  }

  Future<void> _takePicture() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.camera);
      if (image != null) {
        addUploadedImage(image.path);
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('사진을 찍는 중 오류가 발생했습니다: $e')),
      );
    }
  }

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        addUploadedImage(image.path);
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('이미지를 불러오는 중 오류가 발생했습니다: $e')),
      );
    }
  }

  void addUploadedImage(String imagePath) {
    if (uploadedImages.length < 3) {
      setState(() {
        uploadedImages.add(imagePath);
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('최대 3장까지만 업로드할 수 있습니다.')),
      );
    }
  }

  void removeUploadedImage(int index) {
    setState(() {
      uploadedImages.removeAt(index);
    });
  }

  void _resetState() {
    setState(() {
      uploadedImages.clear(); // 업로드된 이미지 목록 초기화
      _detectionResult = null; // 검출 결과 초기화
      _showResult = false; // 결과 화면 표시 상태 초기화
    });
  }

  Future<void> detectObjects() async {
    if (uploadedImages.isEmpty) {
      print('No images uploaded');
      return;
    }
    print('Detecting objects in ${uploadedImages.last}');
    var result =
        await _detector.runInference(File(uploadedImages.last)); // YOLOX 검출 실행
    print('Detection completed');
    setState(() {
      _detectionResult = result; // 검출 결과 저장
      _showResult = true; // 결과 화면 표시 상태 변경
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('YOLOX App')),
      body: Column(
        children: [
          Expanded(
              flex: 2,
              child: _showResult
                  ? CustomPaint(
                      // 검출 결과 표시
                      painter: ImagePainter(_detectionResult!),
                      size: Size.infinite,
                    )
                  : GridView.builder(
                      // 이미지 업로드 그리드 표시
                      padding: const EdgeInsets.all(8),
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3,
                        crossAxisSpacing: 8,
                        mainAxisSpacing: 8,
                      ),
                      itemCount: 3,
                      itemBuilder: (context, index) {
                        return AspectRatio(
                          aspectRatio: 1,
                          child: Container(
                            decoration: BoxDecoration(
                              border: Border.all(
                                  color: Colors.blueGrey[700]!, width: 2),
                            ),
                            child: index < uploadedImages.length
                                ? Stack(
                                    fit: StackFit.expand,
                                    children: [
                                      Image.file(
                                        File(uploadedImages[index]),
                                        fit: BoxFit.cover,
                                      ),
                                      Positioned(
                                        top: 0,
                                        right: 0,
                                        child: GestureDetector(
                                          onTap: () =>
                                              removeUploadedImage(index),
                                          child: Container(
                                            padding: const EdgeInsets.all(2),
                                            decoration: BoxDecoration(
                                              color:
                                                  Colors.black.withOpacity(0.5),
                                              shape: BoxShape.circle,
                                            ),
                                            child: Icon(Icons.close,
                                                size: 20, color: Colors.white),
                                          ),
                                        ),
                                      ),
                                    ],
                                  )
                                : const Icon(Icons.add_photo_alternate,
                                    size: 50, color: Colors.blueGrey),
                          ),
                        );
                      },
                    )),
          Expanded(
            flex: 3,
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (!_showResult) ...[
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('찍어서 올리기'),
                          onPressed: uploadedImages.length < 3
                              ? () {
                                  HapticFeedback.mediumImpact();
                                  _takePicture();
                                }
                              : null,
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 20, vertical: 12),
                            textStyle: const TextStyle(fontSize: 16),
                          ),
                        ),
                        ElevatedButton.icon(
                          icon: const Icon(Icons.photo_library),
                          label: const Text('앨범에서 업로드'),
                          onPressed: uploadedImages.length < 3
                              ? () {
                                  HapticFeedback.mediumImpact();
                                  _pickImage();
                                }
                              : null,
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 20, vertical: 12),
                            textStyle: const TextStyle(fontSize: 16),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 20),
                    ElevatedButton.icon(
                      icon: const Icon(Icons.search),
                      label: const Text('객체 검출'),
                      onPressed:
                          uploadedImages.isNotEmpty ? detectObjects : null,
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size(double.infinity, 50),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        textStyle: const TextStyle(fontSize: 18),
                      ),
                    ),
                  ] else ...[
                    ElevatedButton.icon(
                      icon: const Icon(Icons.refresh),
                      label: const Text('다시하기'),
                      onPressed: _resetState,
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size(double.infinity, 50),
                        padding: const EdgeInsets.symmetric(vertical: 12),
                        textStyle: const TextStyle(fontSize: 18),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class ImagePainter extends CustomPainter {
  final ui.Image image;

  ImagePainter(this.image);

  @override
  void paint(Canvas canvas, Size size) {
    paintImage(
      canvas: canvas,
      rect: Rect.fromLTWH(0, 0, size.width, size.height),
      image: image,
      fit: BoxFit.contain,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}

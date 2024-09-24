import 'dart:io';
import 'dart:ui' as ui;
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:flutter/services.dart' show rootBundle;

import 'colors.dart';

class YOLOXDetector {
  late tfl.Interpreter _interpreter;
  bool _isInitialized = false;
  late List<String> _labels;
  final List<int> inputShape = [640, 640];
  final List<List<double>> _colors = List<List<double>>.generate(
    (CustomColors.length / 3).floor(),
    (index) => CustomColors.sublist(index * 3, (index + 1) * 3),
  );

  YOLOXDetector() {
    _initializeInterpreter();
    _loadLabels();
  }

  Future<void> _initializeInterpreter() async {
    try {
      _interpreter =
          await tfl.Interpreter.fromAsset('assets/yolox_base.tflite');
      _isInitialized = true;
      print('Interpreter initialized successfully');

      // 입력 텐서 정보 출력
      var inputShape = _interpreter.getInputTensor(0).shape;
      print('Model Input tensor shape: $inputShape');

      // 출력 텐서 정보 출력
      var outputShape = _interpreter.getOutputTensor(0).shape;
      print('Model Output tensor shape: $outputShape');
    } catch (e) {
      print('Failed to initialize interpreter: $e');
    }
  }

  Future<void> _loadLabels() async {
    try {
      String labelsData = await rootBundle.loadString('assets/coco_labels.txt');
      _labels = labelsData.split('\n').where((s) => s.isNotEmpty).toList();
      print('Labels loaded successfully: ${_labels.length} labels');
      print('Labels: $_labels'); // 레이블 내용 출력
    } catch (e) {
      print('Failed to load labels: $e');
      _labels = [];
    }
  }

  Future<ui.Image> runInference(File imageFile) async {
    if (!_isInitialized) {
      await _initializeInterpreter();
    }
    try {
      // 이미지 로드 및 전처리
      var image = img.decodeImage(await imageFile.readAsBytes())!;
      var (preprocessedImage, ratio) = _preprocess(image);

      // 추론 실행
      var output = await _runModelInference(preprocessedImage);

      // 후처리 및 NMS
      var predictions =
          _postprocess(output, [image.width, image.height], ratio);

      print('num predictions: ${predictions.length}');

      var dets = _multiclassNms(predictions, 0.45, 0.1);

      print('num dets: ${dets.length}');
      print('dets: $dets');

      // 결과 시각화
      return _visualizeDetections(image, dets);
    } catch (e) {
      print('Error during inference: $e');
      rethrow;
    }
  }

  (Float32List, double) _preprocess(img.Image image) {
    final inputSize = [640, 640];
    // 이미지 리사이징 비율 계산
    var ratio =
        math.min(inputSize[0] / image.width, inputSize[1] / image.height);
    var nw = (image.width * ratio).round();
    var nh = (image.height * ratio).round();

    // 이미지 리사이징
    var resized = img.copyResize(image, width: nw, height: nh);
    var padded = img.Image(width: inputSize[0], height: inputSize[1]);

    // 패딩 적용 (회색으로 채우기)
    img.fill(padded, color: img.ColorRgb8(114, 114, 114));

    // Paste resized image
    for (var y = 0; y < resized.height; y++) {
      for (var x = 0; x < resized.width; x++) {
        padded.setPixel(x, y, resized.getPixel(x, y));
      }
    }

    var input = Float32List(1 * 3 * inputSize[0] * inputSize[1]);
    var index = 0;
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < inputSize[1]; y++) {
        for (var x = 0; x < inputSize[0]; x++) {
          var pixel = padded.getPixel(x, y);
          input[index++] =
              (c == 0 ? pixel.r : (c == 1 ? pixel.g : pixel.b)).toDouble();
        }
      }
    }

    return (input, ratio);
  }

  Future<List<List<List<double>>>> _runModelInference(
      Float32List preprocessedImage) async {
    var inputShape = _interpreter.getInputTensor(0).shape;
    var outputShape = _interpreter.getOutputTensor(0).shape;

    var output = List.generate(
      outputShape[0],
      (_) => List.generate(
        outputShape[1],
        (_) => List<double>.filled(outputShape[2], 0.0),
      ),
    );

    // 추론 실행
    try {
      _interpreter.run(preprocessedImage.reshape(inputShape), output);
    } catch (e) {
      print('Error during inference: $e');
      rethrow;
    }

    return output;
  }

  List<List<double>> _postprocess(
      List<List<List<double>>> modelOutput, List<int> imageSize, double ratio) {
    var predictions = _decodeModelOutput(modelOutput);

    for (var pred in predictions) {
      if (pred.length < 5) continue;
      var boxData = pred.sublist(0, 4);
      var objectnessScore = pred[4];
      var classScores = pred.sublist(5);

      var maxScore = classScores.reduce(math.max);
      var finalScore = objectnessScore * maxScore;

      var x1 = boxData[0] / ratio;
      var y1 = boxData[1] / ratio;
      var x2 = boxData[2] / ratio;
      var y2 = boxData[3] / ratio;

      pred[0] = x1;
      pred[1] = y1;
      pred[2] = x2;
      pred[3] = y2;
      pred[4] = finalScore;
    }

    return predictions;
  }

  List<List<double>> _decodeModelOutput(List<List<List<double>>> modelOutput) {
    final List<int> strides = [8, 16, 32];
    final List<List<double>> predictions = [];

    List<List<List<double>>> grids = [];
    List<List<List<double>>> expandedStrides = [];

    for (int stride in strides) {
      final int gridHeight = inputShape[0] ~/ stride;
      final int gridWidth = inputShape[1] ~/ stride;

      List<List<double>> grid = [];
      for (int y = 0; y < gridHeight; y++) {
        for (int x = 0; x < gridWidth; x++) {
          grid.add([x.toDouble(), y.toDouble()]);
        }
      }
      grids.add(grid);

      List<List<double>> expandedStride = List.generate(
        gridHeight * gridWidth,
        (_) => [stride.toDouble()],
      );
      expandedStrides.add(expandedStride);
    }

    // 모든 feature map의 그리드와 스트라이드 결합
    var concatenatedGrids = grids.expand((e) => e).toList();
    var concatenatedStrides = expandedStrides.expand((e) => e).toList();

    for (int i = 0; i < modelOutput[0].length; i++) {
      var output = modelOutput[0][i];
      var grid = concatenatedGrids[i];
      var stride = concatenatedStrides[i][0];

      // 바운딩 박스 좌표 디코딩
      double x = (output[0] + grid[0]) * stride;
      double y = (output[1] + grid[1]) * stride;
      double w = math.exp(output[2]) * stride;
      double h = math.exp(output[3]) * stride;

      predictions.add([
        x - w / 2, // x1
        y - h / 2, // y1
        x + w / 2, // x2
        y + h / 2, // y2
        output[4], // objectness score
        ...output.sublist(5), // class scores
      ]);
    }

    return predictions;
  }

  List<Map<String, dynamic>> _multiclassNms(
      List<List<double>> predictions, double nmsThr, double scoreThr) {
    var finalDets = <Map<String, dynamic>>[];
    for (var cls = 0; cls < _labels.length; cls++) {
      var clsScores = predictions.map((p) => p[4] * p[5 + cls]).toList();

      var validScoreMask = clsScores.map((score) => score > scoreThr).toList();

      var validIndices = List<int>.generate(validScoreMask.length, (i) => i)
          .where((i) => validScoreMask[i])
          .toList();

      if (validIndices.isEmpty) {
        continue;
      }

      var validBoxes =
          validIndices.map((i) => predictions[i].sublist(0, 4)).toList();
      var validScores = validIndices.map((i) => clsScores[i]).toList();
      var keep = _nms(validBoxes, validScores, nmsThr);

      for (var i in keep) {
        finalDets.add({
          'bbox': validBoxes[i],
          'score': validScores[i],
          'class': _labels[cls],
        });
      }
    }
    return finalDets;
  }

  List<int> _nms(
      List<List<double>> boxes, List<double> scores, double nmsThreshold) {
    var indexList = List<int>.generate(scores.length, (i) => i);
    indexList.sort((a, b) => scores[b].compareTo(scores[a]));

    var keep = <int>[];
    while (indexList.isNotEmpty) {
      var currentIndex = indexList[0];
      keep.add(currentIndex);

      indexList = indexList.where((idx) {
        if (idx == currentIndex) return false;

        var iou = _calculateIoU(boxes[currentIndex], boxes[idx]);
        return iou <= nmsThreshold;
      }).toList();
    }

    return keep;
  }

  double _calculateIoU(List<double> boxA, List<double> boxB) {
    var xA = math.max(boxA[0], boxB[0]);
    var yA = math.max(boxA[1], boxB[1]);
    var xB = math.min(boxA[2], boxB[2]);
    var yB = math.min(boxA[3], boxB[3]);

    var interArea = math.max(0, xB - xA) * math.max(0, yB - yA);
    var boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    var boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    return interArea / (boxAArea + boxBArea - interArea);
  }

  Future<ui.Image> _visualizeDetections(
      img.Image originalImage, List<Map<String, dynamic>> detections) async {
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);

    final ui.Image image = await _convertImgImageToUiImage(originalImage);
    canvas.drawImage(image, ui.Offset.zero, ui.Paint());

    final double strokeWidth = 3.0;
    final double fontSize = 16.0;
    final double padding = 5.0;

    for (var detection in detections) {
      List<double> box = detection['bbox'];
      String label = detection['class'];
      double score = detection['score'];

      int colorIndex = _labels.indexOf(label) % _colors.length;
      List<double> color = _colors[colorIndex];

      final paint = ui.Paint()
        ..style = ui.PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..color = ui.Color.fromRGBO((color[0] * 255).toInt(),
            (color[1] * 255).toInt(), (color[2] * 255).toInt(), 1.0);

      final rect = ui.Rect.fromLTRB(box[0], box[1], box[2], box[3]);
      canvas.drawRect(rect, paint);

      final backgroundPaint = ui.Paint()
        ..style = ui.PaintingStyle.fill
        ..color = ui.Color.fromRGBO((color[0] * 255).toInt(),
            (color[1] * 255).toInt(), (color[2] * 255).toInt(), 0.7);

      final textSpan = TextSpan(
        text: '$label: ${score.toStringAsFixed(2)}',
        style: TextStyle(
          color: Colors.white,
          fontSize: fontSize,
          fontWeight: FontWeight.bold,
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final textWidth = textPainter.width + padding * 2;
      final textHeight = textPainter.height + padding * 2;

      final textBackgroundRect = ui.Rect.fromLTWH(
        box[0],
        box[1],
        textWidth,
        textHeight,
      );

      canvas.drawRect(textBackgroundRect, backgroundPaint);

      textPainter.paint(
        canvas,
        Offset(box[0] + padding, box[1] + padding),
      );
    }

    final picture = recorder.endRecording();
    return picture.toImage(originalImage.width, originalImage.height);
  }

  Future<ui.Image> _convertImgImageToUiImage(img.Image image) async {
    var pngBytes = img.encodePng(image);
    ui.Codec codec = await ui.instantiateImageCodec(pngBytes);
    ui.FrameInfo frameInfo = await codec.getNextFrame();
    return frameInfo.image;
  }
}

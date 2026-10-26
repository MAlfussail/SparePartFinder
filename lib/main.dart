import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SparePartsMatcherApp());
}

class SparePartsMatcherApp extends StatelessWidget {
  const SparePartsMatcherApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'مطابق قطع الغيار الذكي',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorSchemeSeed: const Color(0xFFF59E0B),
        scaffoldBackgroundColor: const Color(0xFF0F172A),
        fontFamily: 'Roboto',
      ),
      builder: (context, child) {
        return Directionality(
          textDirection: TextDirection.rtl,
          child: child!,
        );
      },
      home: const HomeScreen(),
    );
  }
}

/// نموذج يمثل قطعة غيار داخل المجلد المرفوع مع بصمتها البصرية المقاومة للزوايا والإضاءة
class SparePartItem {
  final String filePath;
  final String fileName;
  final String partName;
  final Uint8List bytes;
  final PartFeatureVector features;

  SparePartItem({
    required this.filePath,
    required this.fileName,
    required this.partName,
    required this.bytes,
    required this.features,
  });
}

/// بصمة زاوية واحدة ضمن بنك الزوايا المتعددة
class PoseSignature {
  final int angleDeg;
  final List<double> edgeHist; // 128D Sobel HOG
  final List<int> dHash;       // 64-bit Illumination-Normalized dHash

  PoseSignature({
    required this.angleDeg,
    required this.edgeHist,
    required this.dHash,
  });
}

/// المتجه البصري المتقدم المستخرج بالذكاء الاصطناعي المحلي (مقاوم لزوايا التصوير وتغيرات الإضاءة)
class PartFeatureVector {
  final List<double> chromaHist;     // 32D توزيع لوني معزول عن شدة الإضاءة (Gray-World + HSV Chromaticity)
  final List<double> polarRings;     // 48D حلقات قطبية متمركزة حول مركز ثقل القطعة (مقاومة للدوران 360°)
  final List<double> relAngleHist;   // 16D زوايا الحواف النسبية للشعاع المركزي (ثابتة عند أي زاوية دوران)
  final List<double> fourierMag;     // 9D طيف فورييه الدوراني
  final List<PoseSignature> poseBank; // بنك الزوايا المتعددة (0°, ±20°, ±35°, 90°, 180°, 270°)

  PartFeatureVector({
    required this.chromaHist,
    required this.polarRings,
    required this.relAngleHist,
    required this.fourierMag,
    required this.poseBank,
  });
}

/// نتيجة التطابق لقطعة واحدة
class MatchResult {
  final SparePartItem item;
  final double similarityPercent;
  final double shapeScore;
  final double structureScore;
  final double colorScore;

  MatchResult({
    required this.item,
    required this.similarityPercent,
    required this.shapeScore,
    required this.structureScore,
    required this.colorScore,
  });
}

enum AiMode { offlineLocal, onlineCloud }

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  static const List<int> _canonicalAngles = [0, 18, -18, 35, -35, 90, -90, 180];

  final List<SparePartItem> _catalog = [];
  final ImagePicker _picker = ImagePicker();

  AiMode _aiMode = AiMode.offlineLocal;
  String _geminiApiKey = '';
  bool _isIndexing = false;
  int _indexingProgress = 0;
  int _indexingTotal = 0;

  Uint8List? _queryImageBytes;
  String? _queryImageName;
  bool _isComparing = false;
  int _lastCompareMs = 0;
  List<MatchResult> _topMatches = [];
  String? _statusMessage;

  @override
  void initState() {
    super.initState();
    _loadSavedSettings();
  }

  Future<void> _loadSavedSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _geminiApiKey = prefs.getString('gemini_api_key') ?? '';
      final savedMode = prefs.getString('ai_mode') ?? 'offline';
      _aiMode = savedMode == 'online' ? AiMode.onlineCloud : AiMode.offlineLocal;
    });
  }

  Future<void> _saveSettings(AiMode mode, String apiKey) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('ai_mode', mode == AiMode.onlineCloud ? 'online' : 'offline');
    await prefs.setString('gemini_api_key', apiKey.trim());
  }

  /// 1. اختيار مجلد صور قطع الغيار بالكامل من الهاتف
  Future<void> _pickSparePartsFolder() async {
    try {
      final String? selectedDir = await FilePicker.platform.getDirectoryPath(
        dialogTitle: 'اختر مجلد صور قطع الغيار',
      );
      if (selectedDir == null) return;

      final dir = Directory(selectedDir);
      final allowedExt = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'};
      final files = dir
          .listSync(recursive: true)
          .whereType<File>()
          .where((f) => allowedExt.contains(p.extension(f.path).toLowerCase()))
          .toList();

      if (files.isEmpty) {
        _showSnack('لم يتم العثور على صور داخل المجلد المختار');
        return;
      }

      await _indexFiles(files);
    } catch (e) {
      _showSnack('تعذر قراءة المجلد مباشرة، يمكنك اختيار الصور المتعددة بدلاً من ذلك: $e');
    }
  }

  /// بديل إضافي: اختيار مجموعة صور دفعة واحدة من الهاتف
  Future<void> _pickMultiplePartImages() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: true,
    );
    if (result == null || result.files.isEmpty) return;

    final files = result.files
        .where((f) => f.path != null)
        .map((f) => File(f.path!))
        .toList();

    await _indexFiles(files);
  }

  Future<void> _indexFiles(List<File> files) async {
    setState(() {
      _isIndexing = true;
      _indexingProgress = 0;
      _indexingTotal = files.length;
      _catalog.clear();
      _topMatches.clear();
    });

    final sw = Stopwatch()..start();
    final List<SparePartItem> loaded = [];

    for (int i = 0; i < files.length; i++) {
      final file = files[i];
      try {
        final bytes = await file.readAsBytes();
        final features = _extractInvariantFeatures(bytes);
        if (features != null) {
          final fileName = p.basename(file.path);
          final partName = p.basenameWithoutExtension(file.path).replaceAll('_', ' ');
          loaded.add(SparePartItem(
            filePath: file.path,
            fileName: fileName,
            partName: partName,
            bytes: bytes,
            features: features,
          ));
        }
      } catch (_) {}

      setState(() {
        _indexingProgress = i + 1;
      });
      await Future.delayed(const Duration(milliseconds: 1));
    }

    sw.stop();
    setState(() {
      _catalog.addAll(loaded);
      _isIndexing = false;
      _statusMessage = 'تم تحميل وفهرسة ${_catalog.length} قطعة غيار في ${sw.elapsedMilliseconds} مللي ثانية';
    });

    if (_queryImageBytes != null && _catalog.isNotEmpty) {
      await _runComparison();
    }
  }

  /// 2. التقاط صورة للقطعة من الكاميرا أو المعرض
  Future<void> _pickQueryImage(ImageSource source) async {
    if (_catalog.isEmpty) {
      _showSnack('يرجى أولاً تحميل مجلد صور قطع الغيار في الخطوة رقم 1');
      return;
    }

    final XFile? picked = await _picker.pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 90,
    );
    if (picked == null) return;

    final bytes = await picked.readAsBytes();
    setState(() {
      _queryImageBytes = bytes;
      _queryImageName = picked.name;
    });

    await _runComparison();
  }

  /// 3. تشغيل المقارنة السريعة (محلياً مع مقاومة الزوايا والإضاءة أو عبر الإنترنت) وعرض أفضل 5 تطابقات
  Future<void> _runComparison() async {
    if (_queryImageBytes == null || _catalog.isEmpty) return;

    setState(() {
      _isComparing = true;
    });

    final sw = Stopwatch()..start();

    try {
      if (_aiMode == AiMode.onlineCloud && _geminiApiKey.trim().isNotEmpty) {
        final onlineResults = await _runOnlineGeminiComparison(_queryImageBytes!);
        sw.stop();
        setState(() {
          _topMatches = onlineResults;
          _lastCompareMs = sw.elapsedMilliseconds;
          _isComparing = false;
        });
        return;
      }

      // المقارنة بالذكاء الاصطناعي المحلي المتقدم (بدون إنترنت - مقاوم لزوايا التصوير والإضاءة)
      final queryFeatures = _extractInvariantFeatures(_queryImageBytes!);
      if (queryFeatures == null) {
        _showSnack('تعذر تحليل الصورة المرفوعة');
        setState(() => _isComparing = false);
        return;
      }

      final List<MatchResult> scored = [];
      for (final part in _catalog) {
        final scores = _computeInvariantSimilarity(queryFeatures, part.features);
        scored.add(MatchResult(
          item: part,
          similarityPercent: scores['total']!,
          shapeScore: scores['edge']!,
          structureScore: scores['polar']!,
          colorScore: scores['color']!,
        ));
      }

      scored.sort((a, b) => b.similarityPercent.compareTo(a.similarityPercent));
      final top5 = scored.take(5).toList();

      sw.stop();
      setState(() {
        _topMatches = top5;
        _lastCompareMs = sw.elapsedMilliseconds;
        _isComparing = false;
      });
    } catch (e) {
      sw.stop();
      setState(() => _isComparing = false);
      _showSnack('حدث خطأ أثناء المقارنة: $e');
    }
  }

  /// استخراج متجه زاوية منفردة بعد موازنة الإضاءة والظلال (Gray-World + Local Shadow Leveling + Z-Score)
  Map<String, dynamic> _extractSinglePose(img.Image baseDecoded, int angleDeg) {
    final rotated = angleDeg == 0
        ? img.copyResize(baseDecoded, width: 64, height: 64)
        : img.copyResize(img.copyRotate(baseDecoded, angle: angleDeg), width: 64, height: 64);

    // 1. موازنة اللون الأبيض (Gray-World White Balance) لإزالة اصفرار أو ازرقاق الإضاءة
    double sumR = 0, sumG = 0, sumB = 0;
    const int numPx = 64 * 64;
    for (int y = 0; y < 64; y++) {
      for (int x = 0; x < 64; x++) {
        final px = rotated.getPixel(x, y);
        sumR += px.r;
        sumG += px.g;
        sumB += px.b;
      }
    }
    final meanR = math.max(10.0, sumR / numPx);
    final meanG = math.max(10.0, sumG / numPx);
    final meanB = math.max(10.0, sumB / numPx);
    final avgGray = (meanR + meanG + meanB) / 3.0;
    final gainR = avgGray / meanR;
    final gainG = avgGray / meanG;
    final gainB = avgGray / meanB;

    final List<double> normR = List.filled(numPx, 0.0);
    final List<double> normG = List.filled(numPx, 0.0);
    final List<double> normB = List.filled(numPx, 0.0);
    final List<double> lumRaw = List.filled(numPx, 0.0);

    for (int y = 0; y < 64; y++) {
      for (int x = 0; x < 64; x++) {
        final idx = y * 64 + x;
        final px = rotated.getPixel(x, y);
        final r = (px.r * gainR).clamp(0.0, 255.0);
        final g = (px.g * gainG).clamp(0.0, 255.0);
        final b = (px.b * gainB).clamp(0.0, 255.0);
        normR[idx] = r;
        normG[idx] = g;
        normB[idx] = b;
        lumRaw[idx] = 0.299 * r + 0.587 * g + 0.114 * b;
      }
    }

    // 2. معادلة الظلال المكانية (4x4 Block Illumination Leveling) + توحيد التباين المعياري (Z-Score)
    final List<double> blockMeans = List.filled(16, 128.0);
    for (int by = 0; by < 4; by++) {
      for (int bx = 0; bx < 4; bx++) {
        double bSum = 0;
        for (int y = by * 16; y < (by + 1) * 16; y++) {
          for (int x = bx * 16; x < (bx + 1) * 16; x++) {
            bSum += lumRaw[y * 64 + x];
          }
        }
        blockMeans[by * 4 + bx] = math.max(25.0, bSum / 256.0);
      }
    }

    final List<double> gray = List.filled(numPx, 0.0);
    double gSum = 0;
    for (int y = 0; y < 64; y++) {
      final by = (y ~/ 16).clamp(0, 3);
      for (int x = 0; x < 64; x++) {
        final bx = (x ~/ 16).clamp(0, 3);
        final localBg = blockMeans[by * 4 + bx];
        final v = (lumRaw[y * 64 + x] / localBg) * 128.0 * 0.55 + lumRaw[y * 64 + x] * 0.45;
        gray[y * 64 + x] = v;
        gSum += v;
      }
    }

    final gMean = gSum / numPx;
    double gVar = 0;
    for (int i = 0; i < numPx; i++) {
      final d = gray[i] - gMean;
      gVar += d * d;
    }
    final gStd = math.max(8.0, math.sqrt(gVar / numPx));
    for (int i = 0; i < numPx; i++) {
      gray[i] = (((gray[i] - gMean) / gStd) * 46.0 + 128.0).clamp(0.0, 255.0);
    }

    // 3. حساب حواف Sobel ومركز ثقل القطعة (Object Edge-Energy Centroid)
    final List<double> magMap = List.filled(numPx, 0.0);
    final List<double> angMap = List.filled(numPx, 0.0);
    double energySum = 0, cxSum = 0, cySum = 0;

    for (int y = 2; y < 62; y++) {
      for (int x = 2; x < 62; x++) {
        final gx = -gray[(y - 1) * 64 + (x - 1)] +
            gray[(y - 1) * 64 + (x + 1)] -
            2 * gray[y * 64 + (x - 1)] +
            2 * gray[y * 64 + (x + 1)] -
            gray[(y + 1) * 64 + (x - 1)] +
            gray[(y + 1) * 64 + (x + 1)];

        final gy = -gray[(y - 1) * 64 + (x - 1)] -
            2 * gray[(y - 1) * 64 + x] -
            gray[(y - 1) * 64 + (x + 1)] +
            gray[(y + 1) * 64 + (x - 1)] +
            2 * gray[(y + 1) * 64 + x] +
            gray[(y + 1) * 64 + (x + 1)];

        final mag = math.sqrt(gx * gx + gy * gy);
        magMap[y * 64 + x] = mag;
        double angle = math.atan2(gy, gx);
        if (angle < 0) angle += math.pi;
        angMap[y * 64 + x] = angle;

        if (mag > 22.0) {
          energySum += mag;
          cxSum += x * mag;
          cySum += y * mag;
        }
      }
    }

    final cx = energySum > 100.0 ? (cxSum / energySum).clamp(20.0, 44.0) : 32.0;
    final cy = energySum > 100.0 ? (cySum / energySum).clamp(20.0, 44.0) : 32.0;

    double radSqSum = 0;
    for (int y = 2; y < 62; y++) {
      for (int x = 2; x < 62; x++) {
        final m = magMap[y * 64 + x];
        if (m > 22.0) {
          final dx = x - cx;
          final dy = y - cy;
          radSqSum += (dx * dx + dy * dy) * m;
        }
      }
    }
    final rEff = energySum > 100.0
        ? (math.sqrt(radSqSum / energySum) * 1.45).clamp(12.0, 28.0)
        : 22.0;

    // 4. متجه اتجاهات الحواف المكاني (128D Sobel HOG)
    final List<double> edgeHist = List.filled(128, 0.0);
    for (int y = 2; y < 62; y++) {
      for (int x = 2; x < 62; x++) {
        final mag = magMap[y * 64 + x];
        int bin = ((angMap[y * 64 + x] / math.pi) * 8).floor();
        if (bin >= 8) bin = 7;
        final cellX = (x ~/ 16).clamp(0, 3);
        final cellY = (y ~/ 16).clamp(0, 3);
        edgeHist[(cellY * 4 + cellX) * 8 + bin] += mag;
      }
    }
    _normalizeVector(edgeHist);

    // 5. البصمة الهيكلية 64-bit dHash على الصورة الموحدة الإضاءة
    final List<int> dHash = [];
    for (int hy = 0; hy < 8; hy++) {
      for (int hx = 0; hx < 8; hx++) {
        final y0 = hy * 8 + 4;
        final x1 = math.min(63, hx * 7 + 4);
        final x2 = math.min(63, (hx + 1) * 7 + 4);
        dHash.add(gray[y0 * 64 + x1] > gray[y0 * 64 + x2] ? 1 : 0);
      }
    }

    return {
      'gray': gray,
      'normR': normR,
      'normG': normG,
      'normB': normB,
      'magMap': magMap,
      'angMap': angMap,
      'cx': cx,
      'cy': cy,
      'rEff': rEff,
      'edgeHist': edgeHist,
      'dHash': dHash,
    };
  }

  /// المحرك الشامل لاستخراج البصمة المقاومة لزوايا التصوير والإضاءة
  PartFeatureVector? _extractInvariantFeatures(Uint8List imageBytes) {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return null;

    final base = _extractSinglePose(decoded, 0);
    final List<double> gray = base['gray'];
    final List<double> normR = base['normR'];
    final List<double> normG = base['normG'];
    final List<double> normB = base['normB'];
    final List<double> magMap = base['magMap'];
    final List<double> angMap = base['angMap'];
    final double cx = base['cx'];
    final double cy = base['cy'];
    final double rEff = base['rEff'];

    // 1. الحلقات القطبية المتمركزة (48D Concentric Polar Rings - مقاومة للدوران 360°)
    final List<double> polarRings = List.filled(48, 0.0);
    final List<double> ringCounts = List.filled(8, 0.0);
    final List<double> relativeAngleHist = List.filled(16, 0.0);
    final List<double> angularEdgeProfile = List.filled(16, 0.0);
    final List<double> chromaHist = List.filled(32, 0.0);

    for (int y = 2; y < 62; y++) {
      for (int x = 2; x < 62; x++) {
        final idx = y * 64 + x;
        final dx = x - cx;
        final dy = y - cy;
        final dist = math.sqrt(dx * dx + dy * dy);
        final normDist = dist / (rEff * 1.25);
        final ringIdx = math.min(7, (normDist * 8).floor());
        ringCounts[ringIdx] += 1.0;

        final mag = magMap[idx];
        final gVal = gray[idx] / 255.0;
        final rayAngle = math.atan2(dy, dx);
        final gradAngle = angMap[idx];
        final relDiff = math.cos(gradAngle - rayAngle).abs();

        final rBase = ringIdx * 6;
        polarRings[rBase] += gVal;
        polarRings[rBase + 1] += mag / 255.0;
        polarRings[rBase + 2] += (mag / 255.0) * relDiff;
        polarRings[rBase + 3] += (mag / 255.0) * (1.0 - relDiff);

        final r = normR[idx], g = normG[idx], b = normB[idx];
        final rgbSum = r + g + b + 1.0;
        polarRings[rBase + 4] += r / rgbSum;
        polarRings[rBase + 5] += g / rgbSum;

        if (dist > 3.0 && mag > 18.0) {
          final relBin = math.min(15, (relDiff * 16).floor());
          relativeAngleHist[relBin] += mag;
          int sector = (((rayAngle + math.pi) / (2 * math.pi)) * 16).floor().clamp(0, 15);
          angularEdgeProfile[sector] += mag;
        }

        // توزيع لوني HSV معزول عن شدة الإضاءة
        final maxC = math.max(r, math.max(g, b));
        final minC = math.min(r, math.min(g, b));
        final delta = maxC - minC;
        final sat = maxC > 5.0 ? delta / maxC : 0.0;

        if (sat < 0.14) {
          final metalBin = math.min(7, (gVal * 8).floor());
          chromaHist[metalBin] += 1.0;
        } else {
          double hue = 0;
          if (maxC == r) {
            hue = ((g - b) / delta) % 6;
          } else if (maxC == g) {
            hue = (b - r) / delta + 2;
          } else {
            hue = (r - g) / delta + 4;
          }
          hue *= 60.0;
          if (hue < 0) hue += 360.0;
          final hueBin = 8 + math.min(23, ((hue / 360.0) * 24).floor());
          chromaHist[hueBin] += 1.0 + sat;
        }
      }
    }

    for (int r = 0; r < 8; r++) {
      final cnt = math.max(1.0, ringCounts[r]);
      for (int k = 0; k < 6; k++) {
        polarRings[r * 6 + k] /= cnt;
      }
    }
    _normalizeVector(polarRings);
    _normalizeVector(relativeAngleHist);
    _normalizeVector(chromaHist);
    _normalizeVector(angularEdgeProfile);

    // طيف فورييه الدوراني (Rotation-Invariant Fourier Harmonics)
    final List<double> fourierMag = List.filled(9, 0.0);
    for (int k = 0; k <= 8; k++) {
      double re = 0, im = 0;
      for (int n = 0; n < 16; n++) {
        final phi = (2 * math.pi * k * n) / 16.0;
        re += angularEdgeProfile[n] * math.cos(phi);
        im -= angularEdgeProfile[n] * math.sin(phi);
      }
      fourierMag[k] = math.sqrt(re * re + im * im);
    }
    _normalizeVector(fourierMag);

    // بنك الزوايا المتعددة (Multi-Angle Pose Bank)
    final List<PoseSignature> poseBank = [];
    for (final ang in _canonicalAngles) {
      if (ang == 0) {
        poseBank.add(PoseSignature(
          angleDeg: 0,
          edgeHist: base['edgeHist'],
          dHash: base['dHash'],
        ));
      } else {
        final p = _extractSinglePose(decoded, ang);
        poseBank.add(PoseSignature(
          angleDeg: ang,
          edgeHist: p['edgeHist'],
          dHash: p['dHash'],
        ));
      }
    }

    return PartFeatureVector(
      chromaHist: chromaHist,
      polarRings: polarRings,
      relAngleHist: relativeAngleHist,
      fourierMag: fourierMag,
      poseBank: poseBank,
    );
  }

  void _normalizeVector(List<double> vec) {
    double norm = 0.0;
    for (final v in vec) {
      norm += v * v;
    }
    norm = math.sqrt(norm);
    if (norm > 1e-8) {
      for (int i = 0; i < vec.length; i++) {
        vec[i] /= norm;
      }
    }
  }

  Map<String, double> _computeInvariantSimilarity(PartFeatureVector a, PartFeatureVector b) {
    // 1. التشابه القطبي الدائري الثابت تحت أي زاوية دوران (0°-360°)
    final polarSim = _cosineSimilarity(a.polarRings, b.polarRings);
    final relAngSim = _cosineSimilarity(a.relAngleHist, b.relAngleHist);
    final fourierSim = _cosineSimilarity(a.fourierMag, b.fourierMag);
    final rotInvariantScore = (polarSim * 0.50) + (relAngSim * 0.30) + (fourierSim * 0.20);

    // 2. البحث في بنك الزوايا المتعددة لأفضل محاذاة زاوية بين الصورتين
    double bestEdgeSim = 0.0;
    double bestHashSim = 0.0;

    final queryPoses = a.poseBank.take(3).toList();
    for (final pa in queryPoses) {
      for (final pb in b.poseBank) {
        final eSim = _cosineSimilarity(pa.edgeHist, pb.edgeHist);
        if (eSim > bestEdgeSim) bestEdgeSim = eSim;

        int sameBits = 0;
        for (int i = 0; i < pa.dHash.length; i++) {
          if (pa.dHash[i] == pb.dHash[i]) sameBits++;
        }
        final hSim = sameBits / pa.dHash.length;
        if (hSim > bestHashSim) bestHashSim = hSim;
      }
    }

    // 3. تشابه اللون المعزول عن تغيرات الإضاءة
    final chromaSim = _cosineSimilarity(a.chromaHist, b.chromaHist);

    final shapeAndAngleSim = (bestEdgeSim * 0.52) + (rotInvariantScore * 0.48);
    final structureAndPolarSim = (rotInvariantScore * 0.55) + (bestHashSim * 0.45);
    final combined = (shapeAndAngleSim * 0.45) + (structureAndPolarSim * 0.31) + (chromaSim * 0.24);
    final calibrated = (math.pow(combined, 0.78) * 100.0).clamp(0.0, 100.0);

    return {
      'total': double.parse(calibrated.toStringAsFixed(1)),
      'edge': double.parse((math.pow(shapeAndAngleSim, 0.8) * 100).toStringAsFixed(1)),
      'polar': double.parse((math.pow(structureAndPolarSim, 0.8) * 100).toStringAsFixed(1)),
      'color': double.parse((math.pow(chromaSim, 0.8) * 100).toStringAsFixed(1)),
    };
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    double dot = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return dot.clamp(0.0, 1.0);
  }

  /// المقارنة عبر الإنترنت باستخدام Gemini Vision API + إعادة ترتيب أفضل المرشحين
  Future<List<MatchResult>> _runOnlineGeminiComparison(Uint8List queryBytes) async {
    final queryFeatures = _extractInvariantFeatures(queryBytes)!;
    final List<MatchResult> localCandidates = _catalog.map((part) {
      final s = _computeInvariantSimilarity(queryFeatures, part.features);
      return MatchResult(
        item: part,
        similarityPercent: s['total']!,
        shapeScore: s['edge']!,
        structureScore: s['polar']!,
        colorScore: s['color']!,
      );
    }).toList();
    localCandidates.sort((a, b) => b.similarityPercent.compareTo(a.similarityPercent));
    final topCandidates = localCandidates.take(math.min(8, localCandidates.length)).toList();

    final List<Map<String, dynamic>> partsPayload = [
      {
        'text': 'أنت نظام ذكاء اصطناعي متخصص في مطابقة صور قطع الغيار حتى عند اختلاف زاوية التصوير والإضاءة. '
            'الصورة الأولى هي صورة القطعة المراد البحث عنها. '
            'الصور التالية مرقمة من 0 إلى ${topCandidates.length - 1} لقطع الغيار المرشحة. '
            'أعد فقط مصفوفة JSON تحتوي على أفضل 5 تطابقات بالصيغة: '
            '[{"index": 0, "score": 96.5}] مرتبة تنازلياً حسب نسبة التطابق البصري الدقيق.'
      },
      {
        'inline_data': {
          'mime_type': 'image/jpeg',
          'data': base64Encode(queryBytes),
        }
      }
    ];

    for (int i = 0; i < topCandidates.length; i++) {
      partsPayload.add({'text': 'CANDIDATE_$i (${topCandidates[i].item.partName}):'});
      partsPayload.add({
        'inline_data': {
          'mime_type': 'image/jpeg',
          'data': base64Encode(topCandidates[i].item.bytes),
        }
      });
    }

    final uri = Uri.parse(
      'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${_geminiApiKey.trim()}',
    );
    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'contents': [
          {'parts': partsPayload}
        ],
        'generationConfig': {'responseMimeType': 'application/json'}
      }),
    );

    if (resp.statusCode == 200) {
      final data = jsonDecode(resp.body);
      final text = data['candidates']?[0]?['content']?['parts']?[0]?['text'] ?? '[]';
      final List<dynamic> parsed = jsonDecode(text);
      final List<MatchResult> results = [];
      for (final entry in parsed) {
        final idx = (entry['index'] as num).toInt();
        final score = (entry['score'] as num).toDouble().clamp(0.0, 100.0);
        if (idx >= 0 && idx < topCandidates.length) {
          final base = topCandidates[idx];
          results.add(MatchResult(
            item: base.item,
            similarityPercent: double.parse(score.toStringAsFixed(1)),
            shapeScore: base.shapeScore,
            structureScore: base.structureScore,
            colorScore: base.colorScore,
          ));
        }
      }
      if (results.isNotEmpty) {
        return results.take(5).toList();
      }
    }
    return topCandidates.take(5).toList();
  }

  void _showSnack(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), behavior: SnackBarBehavior.floating),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E293B),
        title: const Text(
          'مطابق قطع الغيار بالذكاء الاصطناعي',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_suggest),
            tooltip: 'إعدادات الذكاء الاصطناعي',
            onPressed: _openAiSettingsModal,
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildAiModeSelectorCard(),
            const SizedBox(height: 16),
            _buildStep1FolderCard(),
            const SizedBox(height: 16),
            _buildStep2QueryCard(),
            const SizedBox(height: 16),
            _buildStep3ResultsCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildAiModeSelectorCard() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1E293B),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.amber.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'آلية الذكاء الاصطناعي (مقاومة لزوايا التصوير وتغيرات الإضاءة):',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: Colors.amber),
          ),
          const SizedBox(height: 10),
          SegmentedButton<AiMode>(
            segments: const [
              ButtonSegment(
                value: AiMode.offlineLocal,
                icon: Icon(Icons.offline_bolt),
                label: Text('محلي (بدون إنترنت)'),
              ),
              ButtonSegment(
                value: AiMode.onlineCloud,
                icon: Icon(Icons.cloud_done),
                label: Text('سحابي (بإنترنت)'),
              ),
            ],
            selected: {_aiMode},
            onSelectionChanged: (val) {
              setState(() => _aiMode = val.first);
              _saveSettings(_aiMode, _geminiApiKey);
              if (_queryImageBytes != null && _catalog.isNotEmpty) {
                _runComparison();
              }
            },
          ),
        ],
      ),
    );
  }

  Widget _buildStep1FolderCard() {
    return Card(
      color: const Color(0xFF1E293B),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const CircleAvatar(
                  radius: 14,
                  backgroundColor: Colors.amber,
                  child: Text('1', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
                ),
                const SizedBox(width: 10),
                const Expanded(
                  child: Text(
                    'تحميل مجلد صور قطع الغيار',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.amber.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '${_catalog.length} قطعة',
                    style: const TextStyle(color: Colors.amber, fontWeight: FontWeight.bold),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isIndexing ? null : _pickSparePartsFolder,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('رفع مجلد كامل'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.amber,
                      foregroundColor: Colors.black,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isIndexing ? null : _pickMultiplePartImages,
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('اختيار صور متعددة'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
              ],
            ),
            if (_isIndexing) ...[
              const SizedBox(height: 12),
              LinearProgressIndicator(
                value: _indexingTotal > 0 ? _indexingProgress / _indexingTotal : null,
              ),
              const SizedBox(height: 6),
              Text('جاري تحليل وفهرسة الصور: $_indexingProgress / $_indexingTotal'),
            ],
            if (_statusMessage != null) ...[
              const SizedBox(height: 8),
              Text(_statusMessage!, style: const TextStyle(color: Colors.greenAccent, fontSize: 12)),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildStep2QueryCard() {
    return Card(
      color: const Color(0xFF1E293B),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                CircleAvatar(
                  radius: 14,
                  backgroundColor: Colors.amber,
                  child: Text('2', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
                ),
                SizedBox(width: 10),
                Text(
                  'رفع صورة القطعة للمقارنة',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickQueryImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('التقاط بالكاميرا'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF3B82F6),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickQueryImage(ImageSource.gallery),
                    icon: const Icon(Icons.image_search),
                    label: const Text('من معرض الصور'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF10B981),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
                ),
              ],
            ),
            if (_queryImageBytes != null) ...[
              const SizedBox(height: 14),
              Row(
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Image.memory(
                      _queryImageBytes!,
                      width: 80,
                      height: 80,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('الصورة قيد الفحص (مع توحيد الإضاءة والزوايا):', style: TextStyle(color: Colors.white70, fontSize: 12)),
                        Text(
                          _queryImageName ?? 'صورة ملتقطة',
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'زمن المقارنة: $_lastCompareMs مللي ثانية',
                          style: const TextStyle(color: Colors.amber, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildStep3ResultsCard() {
    return Card(
      color: const Color(0xFF1E293B),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                CircleAvatar(
                  radius: 14,
                  backgroundColor: Colors.amber,
                  child: Text('3', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
                ),
                SizedBox(width: 10),
                Text(
                  'أفضل 5 تطابقات (Top 5 Matches)',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (_isComparing)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(24),
                  child: CircularProgressIndicator(),
                ),
              )
            else if (_topMatches.isEmpty)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 20),
                child: Center(
                  child: Text(
                    'قم برفع مجلد قطع الغيار ثم اختر صورة من الكاميرا أو المعرض لعرض أفضل 5 تطابقات هنا.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white60),
                  ),
                ),
              )
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: _topMatches.length,
                separatorBuilder: (_, __) => const Divider(color: Colors.white12, height: 18),
                itemBuilder: (context, index) {
                  final match = _topMatches[index];
                  final isFirst = index == 0;
                  return Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: isFirst ? Colors.amber.withOpacity(0.1) : Colors.black26,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: isFirst ? Colors.amber : Colors.white10,
                        width: isFirst ? 1.5 : 1,
                      ),
                    ),
                    child: Row(
                      children: [
                        Container(
                          width: 32,
                          height: 32,
                          alignment: Alignment.center,
                          decoration: BoxDecoration(
                            color: isFirst ? Colors.amber : Colors.white12,
                            shape: BoxShape.circle,
                          ),
                          child: Text(
                            '#${index + 1}',
                            style: TextStyle(
                              color: isFirst ? Colors.black : Colors.white,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.memory(
                            match.item.bytes,
                            width: 64,
                            height: 64,
                            fit: BoxFit.cover,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                match.item.partName,
                                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                              ),
                              Text(
                                'الملف: ${match.item.fileName}',
                                style: const TextStyle(color: Colors.white54, fontSize: 11),
                              ),
                              const SizedBox(height: 6),
                              ClipRRect(
                                borderRadius: BorderRadius.circular(6),
                                child: LinearProgressIndicator(
                                  value: match.similarityPercent / 100.0,
                                  minHeight: 7,
                                  backgroundColor: Colors.white12,
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    match.similarityPercent >= 80
                                        ? Colors.greenAccent
                                        : match.similarityPercent >= 60
                                            ? Colors.amber
                                            : Colors.orangeAccent,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 12),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            Text(
                              '${match.similarityPercent.toStringAsFixed(1)}%',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 17,
                                color: isFirst ? Colors.amber : Colors.greenAccent,
                              ),
                            ),
                            const Text(
                              'نسبة التطابق',
                              style: TextStyle(fontSize: 10, color: Colors.white54),
                            ),
                          ],
                        ),
                      ],
                    ),
                  );
                },
              ),
          ],
        ),
      ),
    );
  }

  void _openAiSettingsModal() {
    final controller = TextEditingController(text: _geminiApiKey);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF1E293B),
        title: const Text('إعدادات الذكاء الاصطناعي السحابي (Gemini API)'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'الوضع المحلي يعمل مجاناً وبدون إنترنت 100% مع مقاومة زوايا التصوير والإضاءة. إذا أردت تفعيل وضع الإنترنت السحابي عبر Gemini Vision، أدخل مفتاح API الاختياري هنا:',
              style: TextStyle(fontSize: 13, color: Colors.white70),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                labelText: 'مفتاح Gemini API Key',
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('إلغاء'),
          ),
          ElevatedButton(
            onPressed: () {
              setState(() => _geminiApiKey = controller.text.trim());
              _saveSettings(_aiMode, _geminiApiKey);
              Navigator.pop(ctx);
            },
            child: const Text('حفظ'),
          ),
        ],
      ),
    );
  }
}

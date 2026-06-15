"""
prompts.py
==========
Verbatim prompt artifacts from the research notebooks for
"Generating Mind Maps from Textual Content: Multilingual Text Processing
and Evaluation Metrics with Large Language Models."

The mind-map system prompts (EN / TR / AR) and the pipeline / critic prompts
below are reproduced exactly as used in the experiments. Only a small, clean
public API (builders + registries) has been added at the bottom; the prompt
text itself is unchanged.

Three critics:
  1. Local Structure Critic   -> are all leaf values specific (not abstract)?
  2. Global Structure Critic  -> are the paths/TOC informative (not generic)?
  3. Factual Critic           -> is every path grounded in a source sentence?
"""


# =============================================================================
# ENGLISH PROMPT
# =============================================================================

ENGLISH_PROMPT = '''# AUTOMATIC ITERATIVE MIND MAP GENERATOR v2.0
## Based on Jain et al. (2024) Methodology

You are an expert knowledge extraction system that transforms text into structured mind maps.
Execute ALL steps automatically without user interaction until completion.
Please focus on all information mentioned in the input text.

───────────────────────────────────────────────────

## CORE PRINCIPLES

1. **Complete Coverage**: Every significant fact in the text MUST appear in the map
2. **Text Fidelity**: Only information explicitly stated in the text
3. **No Redundancy**: Same concept cannot appear in multiple nodes
4. **Logical Hierarchy**: Parent-child relationships must be meaningful
5. **Canonical Uniqueness**: Each node has a unique lemmatized identifier

───────────────────────────────────────────────────

## ⭐ COVERAGE-FIRST DIRECTIVE (applies equally to every model)

1. **Completeness outranks everything else** — brevity, balance, symmetry, and the
   "7±2 children" guideline are preferences, never a reason to drop a fact. If any
   rule would remove a fact that is in the text, KEEP THE FACT.
2. **Exhaustive extraction** — internally scan the text sentence by sentence and list
   every name, date, number, place, event, and relationship. Nothing on that list may
   be missing from the final map.
3. **Expand until nothing remains** — after each pass ask: "Is any fact from the text
   not yet a node?" If yes, add it and repeat. Stop only when the text is fully covered.
4. **Final coverage audit** — before output, re-read every sentence and confirm all of
   its facts appear in the map; add any that are missing.
5. **Never summarise, compress, or merge** distinct facts into a vague node.
6. **Do not shorten the map to look "clean."** A faithful map of an ordinary paragraph
   usually has many nodes (often 15–40+). When unsure, add the node — prefer too many
   over too few. A short, tidy map is a FAILURE if any fact is missing.

───────────────────────────────────────────────────

## ⛔ STRICT PROHIBITIONS

1. **No Paraphrasing Names**: Keep all names, titles, and proper nouns EXACTLY as they appear in the text
   - ❌ "Robert Smith" → "R. Smith" or "Smith"
   - ✅ "Robert Smith" → "Robert Smith"

2. **No Invented Nodes**: Do NOT create nodes for concepts not explicitly mentioned in the text
   - ❌ Adding "Background" or "Overview" if not in text
   - ✅ Only nodes with direct textual evidence

3. **No Merging**: Do NOT combine separate concepts into one node
   - ❌ "Economic and Social" as one node
   - ✅ "Economic" and "Social" as separate nodes

───────────────────────────────────────────────────

## PHASE 0: INTERNAL PRE-ANALYSIS (No Output)

Before generating, internally:
□ Identify all entities, concepts, and key terms
□ Map relationships between entities
□ Create checklist of ALL facts to be captured
□ Plan logical hierarchy structure

───────────────────────────────────────────────────

## HIERARCHY STRUCTURE

| Level | Purpose | Content Type | Examples |
|-------|---------|--------------|----------|
| **Root (L0)** | Central topic | Main subject | "Climate Change", "Napoleon Bonaparte" |
| **L1** | Major themes | Category headers | "Causes", "Effects", "Biography" |
| **L2** | Sub-categories | Grouped details | "Economic Causes", "Environmental Effects" |
| **L3** | Specific facts | Concrete information | "CO2 increased 40%", "Born 1769" |
| **L4** | Supporting details | Only if text supports | Dates, quotes, statistics |

### Branching Guidelines:
- **MECE Principle**: Mutually Exclusive, Collectively Exhaustive
- **Balance**: Avoid asymmetric trees (one deep branch, others shallow)
- **7±2 Rule**: Each parent ideally has 2-9 children
- **Depth Limit**: Go deeper only when text explicitly supports it

───────────────────────────────────────────────────

## EXPANDED RELATIONSHIP TAXONOMY

### Hierarchical:
| Relation | Meaning | Language Indicators |
|----------|---------|---------------------|
| `is_a` | X is a type of Y | "is a", "type of", "kind of", "classified as" |
| `part_of` | X is component of Y | "part of", "component", "consists of", "includes" |
| `instance_of` | X is example of Y | "for example", "such as", "e.g.", "including" |

### Descriptive:
| Relation | Meaning | Language Indicators |
|----------|---------|---------------------|
| `has_property` | X has characteristic Y | "is", "has", "characterized by", adjectives |
| `has_value` | X has quantity Y | numbers, measurements, statistics |
| `contrasts_with` | X differs from Y | "but", "however", "unlike", "whereas" |

### Causal:
| Relation | Meaning | Language Indicators |
|----------|---------|---------------------|
| `causes` | X leads to Y | "causes", "leads to", "results in", "produces" |
| `caused_by` | X results from Y | "because", "due to", "as a result of" |
| `enables` | X makes Y possible | "allows", "enables", "permits", "facilitates" |

### Contextual:
| Relation | Meaning | Language Indicators |
|----------|---------|---------------------|
| `located_in` | X exists in place Y | "in", "at", "located", place names |
| `time_of` | X occurs at time Y | "when", "during", dates, time expressions |
| `agent_of` | X performs action Y | subject position, active voice |
| `patient_of` | X receives action Y | object position, passive voice |
| `associated_with` | X relates to Y | "related to", "connected with", "involves" |

───────────────────────────────────────────────────

## MORPHOLOGICAL PROCESSING

### Normalization Pipeline:
Input Text → Unicode NFC → Lowercase (locale-aware) → Remove Diacritics → Lemmatize

### Node Structure:
```json
{
  "id": "unique_identifier",
  "canonical": "lemmatized_form",
  "label": "Display Label",
  "aliases": ["surface_form_1", "surface_form_2"],
  "relation": "relation_to_parent",
  "evidence": "supporting text snippet",
  "children": []
}
```

### Rules:
- `canonical` = dictionary/lemma form (unique across all nodes)
- `aliases` = all surface forms found in text
- `evidence` = quote or paraphrase proving this node
- `label` = EXACT text as found in source (no paraphrasing)

───────────────────────────────────────────────────

## ITERATIVE EXECUTION PROTOCOL

⚠️ **IMPORTANT: Execute all steps INTERNALLY. Do NOT print intermediate steps or JSON.**

### STEP 1 — Root Selection (internal):
Choose primary concept that is the root

### STEP 2 — Branch Check (internal):
Can we add branches?
□ Unprocessed information remaining?
□ Existing nodes need expansion?
□ Text supports deeper detail?

### STEP 3 — Branch Expansion (internal):
If Yes → Add branches → Return to Step 2
If No → Proceed to output

**Repeat Steps 2-3 internally until complete**

───────────────────────────────────────────────────

## ✓ QUALITY CHECKLIST (Internal)

Before final output, verify:
□ Is there information in the text NOT yet in the map?
□ Is every node supported by actual text content?
□ Are there duplicate or synonymous nodes?
□ Is the hierarchy logical and balanced?
□ Are relationships correctly identified?
□ Is canonical uniqueness maintained?
□ Are all names kept exactly as in text? (No paraphrasing)
□ Are there any invented nodes not in text? (Remove them)
□ Are separate concepts in separate nodes? (No merging)

───────────────────────────────────────────────────

## ERROR PREVENTION

| Error Type | Prevention Method |
|------------|-------------------|
| Missing information | Coverage checklist |
| Hallucination | Evidence requirement for each node |
| Duplication | Canonical uniqueness check |
| Over-branching | MECE + depth justification |
| Wrong relations | Language indicator matching |
| Paraphrased names | Keep label = exact text |
| Invented nodes | Every node needs text evidence |
| Merged concepts | One concept per node |

───────────────────────────────────────────────────

## TASK

Given the text:
{{ input_text }}

───────────────────────────────────────────────────

## OUTPUT INSTRUCTIONS

⚠️ **Do NOT print any intermediate steps, checks, or partial JSON structures.**

Process everything internally, then output ONLY the final complete result in this format:

MindMap
{"root": {"id": "root", "canonical": "...", "label": "...", "children": [...]}}
END_THOUGHT

Begin now. Output ONLY the final MindMap.'''


# =============================================================================
# TURKISH PROMPT
# =============================================================================

TURKISH_PROMPT = '''# OTOMATİK ZİHİN HARİTASI ÜRETİCİSİ

Sen, metinden yapılandırılmış bilgi çıkaran bir zihin haritası sistemisin.
Tüm adımları kullanıcı müdahalesi olmadan otomatik tamamlayacaksın.

## TEMEL İLKELER

1. **Tam Kapsam**: Metindeki HER önemli bilgi haritada yer almalı
2. **Metin Sadakati**: Yalnızca metinde geçen bilgiler eklenebilir
3. **Tekrarsızlık**: Aynı kavram farklı düğümlerde tekrarlanamaz
4. **Mantıksal Hiyerarşi**: Üst-alt ilişkileri anlamlı olmalı

## ⭐ ÖNCELİK: TAM KAPSAM (her model için geçerli)

1. **Tamlık her şeyden önce gelir** — kısalık, denge, simetri ve "7±2 çocuk" kuralı
   yalnızca tercihtir; bir bilgiyi atmak için gerekçe olamaz. Bir kural metindeki bir
   bilgiyi silecekse, BİLGİYİ KORU.
2. **Eksiksiz çıkarım** — metni cümle cümle dahili olarak tara; her ismi, tarihi, sayıyı,
   yeri, olayı ve ilişkiyi listele. Bu listedeki hiçbir şey nihai haritada eksik olamaz.
3. **Hiçbir şey kalmayana kadar genişlet** — her turdan sonra sor: "Metindeki bir bilgi
   henüz düğüm değil mi?" Evetse ekle ve tekrarla. Yalnızca metin tamamen kapsandığında dur.
4. **Son kapsam denetimi** — çıktıdan önce her cümleyi yeniden oku ve tüm bilgilerinin
   haritada olduğunu doğrula; eksik olanları ekle.
5. Farklı bilgileri **asla özetleme, sıkıştırma veya tek bir belirsiz düğümde birleştirme.**
6. **Haritayı "temiz" görünsün diye kısaltma.** Sıradan bir paragrafın sadık haritası
   genellikle çok sayıda düğüm içerir (çoğu zaman 15–40+). Emin değilsen düğümü ekle —
   az yerine çok tercih et. Bir bilgi eksikse, kısa ve düzenli bir harita BAŞARISIZDIR.

## HİYERARŞİ YAPISI

| Seviye | İçerik |
|--------|--------|
| Kök (L0) | Metnin ana konusu |
| L1 | Tematik kategoriler (tanım, bileşenler, türler, özellikler) |
| L2 | Alt gruplar ve sınıflandırmalar |
| L3 | Metinden çıkarılan somut bilgiler |
| L4 | Yalnızca metin destekliyorsa ek detaylar |

## İLİŞKİ TÜRLERİ

- `is_a`: X, Y'nin bir türüdür
- `part_of`: X, Y'nin parçasıdır
- `has_property`: X, Y özelliğine sahiptir
- `enables`: X, Y'yi sağlar/mümkün kılar
- `contrasts_with`: X, Y'den farklıdır
- `associated_with`: X, Y ile ilişkilidir
- `located_in`: X, Y'de bulunur
- `example_of`: X, Y'nin örneğidir

## TÜRKÇE MORFOLOJİ

### 1. Kök/Lemma Dönüşümü
Her kelimeyi kök/lemma formuna dönüştür
Örnek: "araçlardır" → "araç" | "gidiyordu" → "git" | "evlerin" → "ev"

### 2. Hâl Ekleri → İlişki Türleri

| Ek | İsim | İlişki |
|----|------|--------|
| -de / -da / -te / -ta | Bulunma (Locative) | `located_in` |
| -i / -ı / -u / -ü | Belirtme (Accusative) | `patient_of` |
| -in / -ın / -un / -ün | Tamlayan (Genitive) | `part_of` / possessor |
| -e / -a | Yönelme (Dative) | `direction_to` |
| -den / -dan / -ten / -tan | Ayrılma (Ablative) | `source_of` |
| -le / -la / -yle / -yla | Birliktelik (Instrumental) | `associated_with` |

### 3. Çoğul ve İyelik Ekleri

| Ek | Anlam |
|----|-------|
| -ler / -lar | Çoğul |
| -im / -ım / -um / -üm | 1. tekil iyelik |
| -in / -ın / -un / -ün | 2. tekil iyelik |
| -i / -ı / -u / -ü / -si / -sı | 3. tekil iyelik |

### 4. Aliases
Tüm yüzey biçimlerini "aliases" olarak sakla
Örnek: canonical: "ev" → aliases: ["evde", "evden", "eve", "evin", "evler"]

## YİNELEMELİ DÖNGÜ

Aşağıdaki adımları otomatik uygula:

1) **Kök Seç** → Ana kavramı belirle, JSON olarak yaz, `END_THOUGHT`
2) **Dal Eklenebilir mi?** → Metinde işlenmemiş bilgi var mı kontrol et
   - Evet → dalları ekle, JSON güncelle, `END_THOUGHT`
   - Hayır → döngüyü bitir
3) **Tekrarla** → Metin tamamen kapsanana kadar 4. adıma dön

## KALİTE KONTROL

Her iterasyonda şunları doğrula:
□ Metinde henüz eklenmemiş bilgi var mı?
□ Eklenen düğümler gerçekten metinde geçiyor mu?
□ Tekrarlayan veya eş anlamlı düğüm var mı?
□ Hiyerarşi mantıklı mı?

## ÇIKTI KURALLARI

- YALNIZCA geçerli JSON döndür
- ```json``` veya ``` işaretleri KULLANMA
- Açıklama, yorum veya markdown YAZMA
- Yanıtın direkt { ile başlamalı, } ile bitmeli
- JSON dışında HİÇBİR ŞEY yazma

## GÖREV

Metin:
{input_text}

Şimdi kök düğümden başlayarak tüm yinelemeleri otomatik tamamla.
Her adımda END_THOUGHT kullan. Metin tamamen kapsanınca dur.
Yanıtın SADECE JSON olsun:'''


# =============================================================================
# ARABIC PROMPT
# =============================================================================

ARABIC_PROMPT = '''أنت نظام عربي متخصص في إنتاج خرائط ذهنية وفق منهج تكراري (Iterative MindMap Reasoning) شبيه بطريقة Jain et al. 2024.
مطلوب منك تنفيذ جميع الخطوات **تلقائيًا بالكامل** دون أي سؤال للمستخدم ودون انتظار ردود "نعم/لا".
يجب عليك تنفيذ دورة كاملة من الخطوات:
(اختيار الجذر → هل يمكن الإضافة؟ → إضافة فروع → إعادة الفحص → إضافة فروع… إلى أن تصل أنت بنفسك إلى مرحلة "لا يمكن إضافة فروع").

⚠️ تنبيه هام: أجب بالعربية فقط في جميع الشروحات والتعليقات. ممنوع استخدام أي لغة أخرى (إنجليزية، صينية، إلخ).

====================================================
⭐ توجيه التغطية الكاملة (ينطبق على كل النماذج)
====================================================

1. الاكتمال يتقدّم على كل شيء — الاختصار والتوازن والتناظر وقاعدة "7±2 أبناء" مجرد تفضيلات،
   وليست مبرّرًا لإسقاط أي معلومة. إذا كانت قاعدة ستحذف معلومة موجودة في النص، فاحتفظ بالمعلومة.
2. استخراج شامل — امسح النص داخليًا جملةً جملة، وسجّل كل اسم وتاريخ ورقم ومكان وحدث وعلاقة.
   لا يجوز أن يغيب أي عنصر من هذه القائمة عن الخريطة النهائية.
3. وسّع حتى لا يتبقّى شيء — بعد كل تكرار اسأل: "هل بقيت معلومة في النص لم تتحوّل إلى عقدة؟"
   إن وُجدت فأضِفها وكرّر، ولا تتوقّف إلا عند تغطية النص بالكامل.
4. تدقيق التغطية النهائي — قبل الإخراج أعد قراءة كل جملة وتأكّد أن كل معلوماتها ظهرت في الخريطة،
   وأضِف أي ناقص.
5. لا تلخّص أو تضغط أو تدمج معلومات مختلفة في عقدة غامضة أبدًا.
6. لا تختصر الخريطة لتبدو "مرتّبة". الخريطة الأمينة لفقرة عادية تحتوي عادةً على عدد كبير من
   العقد (غالبًا 15–40+). عند الشك أضِف العقدة — فضِّل الكثرة على القِلّة. الخريطة القصيرة
   المرتّبة تُعَدّ فاشلة إذا غابت عنها أي معلومة.

====================================================
(1) شكل إخراج كل خطوة — يجب الالتزام به كما هو
====================================================

1) خطوة اختيار الجذر:
Choose primary concept that is the root
Output:
MindMap
{ JSON للجذر فقط }
END_THOUGHT

2) خطوة سؤال الإضافة:
Can we add branches?
Output: Yes/No
END_THOUGHT

3) خطوة إضافة فروع جديدة:
Add branches:
MindMap
{ JSON الجديد بعد الإضافة }
END_THOUGHT

ويجب عليك تكرار هذا التسلسل تلقائيًا حتى تصل إلى Output: No، ثم تتوقف.

ممنوع دمج الخطوات — ممنوع إهمال END_THOUGHT — ممنوع تخطي أي خطوة.

====================================================
(2) قواعد بناء الخريطة الذهنية
====================================================

● الجذر = المفهوم المركزي للنص.
● المستوى الأول = محاور عامة (الكاتب، النوع، الأحداث، الشخصيات، الأسباب…)
● المستوى الثاني = تصنيفات فرعية منظمة.
● المستوى الثالث = معلومات نصية دقيقة فقط.
● يمكن إضافة مستوى رابع إذا دعمه النص بوضوح.
● ممنوع اختراع معلومات غير موجودة في النص.
● ممنوع التكرار أو إعادة صياغة نفس الفكرة في عقد متعددة.

====================================================
(3) المعايير اللغوية للغة العربية (Morphological Awareness)
====================================================

قبل إنشاء أي عقدة:

1) فصل اللواحق (clitics):
و / ف / ب / ل / ك / س / ال
مثال:
"وبالوقائع" → ["و","ب","ال","وقائع"] → canonical = "واقعة"

2) التطبيع:
● جميع أشكال الألف ← "ا"
● جميع أشكال الياء ← "ي"
● إزالة التشكيل
● تطبيق NFC/NFKC

3) canonical:
● يجب أن يكون الجذر أو اللمّة الصافية للكلمة.
● يجب أن تكون كل canonical فريدة وغير مكررة.

4) aliases:
● كل الأشكال السطحية للكلمة كما ظهرت في النص.

5) العلاقات المسموح بها:
["is_a","part_of","causes","associated_with","agent_of","patient_of","located_in","time_of"]

6) RTL:
● الحفاظ على اتجاه النص من اليمين لليسار في surface.

====================================================
(4) قواعد منع الأخطاء
====================================================

● منع تكرار المحاور
● منع ازدواج الفروع
● منع التشعب الزائد
● منع إضافة عقد غير مفيدة
● منع الهلوسة
● دمج المتشابه
● الحفاظ على بنية هرمية واضحة
● ⚠️ منع استخدام لغات غير العربية في الشرح والتعليقات

====================================================
(5) طريقة التشغيل التلقائي Iterative Loop
====================================================

يجب عليك تنفيذ الخطوات بهذا الترتيب، تلقائيًا، دون تدخل من المستخدم:

1) اختر الجذر (الخطوة 1)
2) نفذ خطوة "Can we add branches?"
3) إذا Yes → نفذ "Add branches"
4) ثم أعد الخطوة رقم 2
5) استمر في هذه الدورة حتى تصل إلى "Output: No"
6) ثم أنهِ الخريطة

كل ذلك في نفس الإخراج، بشكل متتالٍ ومنظم.

====================================================
(6) المبادئ الأساسية
====================================================

1. **التغطية الشاملة**: كل معلومة مهمة في النص يجب أن تظهر في الخريطة
2. **الأمانة النصية**: فقط المعلومات الموجودة فعلياً في النص
3. **عدم التكرار**: لا تُكرر نفس المفهوم في عقد مختلفة
4. **التسلسل المنطقي**: العلاقات بين العقد يجب أن تكون منطقية

====================================================
(7) هيكل المستويات
====================================================

| المستوى | المحتوى | مثال |
|---------|---------|------|
| الجذر (L0) | الموضوع الرئيسي للنص | "الثورة الصناعية" |
| L1 | المحاور الكبرى | "الأسباب"، "النتائج"، "المراحل"، "الشخصيات" |
| L2 | التصنيفات الفرعية | "أسباب اقتصادية"، "أسباب اجتماعية" |
| L3 | المعلومات المحددة من النص | "اختراع المحرك البخاري" |
| L4 | تفاصيل إضافية (إن وُجدت بالنص) | تاريخ، أرقام، اقتباسات |

====================================================
(8) أنواع العلاقات
====================================================

### علاقات هرمية:
| العلاقة | المعنى | المؤشرات اللغوية |
|---------|--------|-----------------|
| `is_a` | X نوع من Y | "هو"، "تُعد"، "يُعتبر" |
| `part_of` | X جزء من Y | "جزء"، "مكوّن"، "يتضمن" |
| `example_of` | X مثال على Y | "مثل"، "كـ"، "منها" |

### علاقات وصفية:
| العلاقة | المعنى | المؤشرات اللغوية |
|---------|--------|-----------------|
| `has_property` | X يمتلك صفة Y | "يتميز بـ"، "له"، "ذو" |
| `has_value` | X له قيمة Y | الأرقام والقياسات |
| `contrasts_with` | X يختلف عن Y | "لكن"، "بينما"، "على عكس" |

### علاقات سببية:
| العلاقة | المعنى | المؤشرات اللغوية |
|---------|--------|-----------------|
| `causes` | X يؤدي إلى Y | "أدى إلى"، "سبّب"، "نتج عنه" |
| `caused_by` | X ناتج عن Y | "بسبب"، "نتيجة"، "جراء" |
| `enables` | X يُمكّن Y | "يسمح بـ"، "يتيح" |

### علاقات أخرى:
| العلاقة | المعنى |
|---------|--------|
| `located_in` | X موجود في Y |
| `time_of` | X حدث في وقت Y |
| `agent_of` | X فاعل Y |
| `patient_of` | X مفعول به لـ Y |
| `associated_with` | X مرتبط بـ Y |

====================================================
(9) المعالجة الصرفية العربية
====================================================

### فصل اللواصق:
"وبالمعلومات" → و + ب + ال + معلومات → canonical: "معلومة"
"فسيكتبونها" → ف + س + يكتبون + ها → canonical: "كتب"

### التطبيع:
- الهمزات: أ إ آ ← ا
- الياء: ى ← ي
- إزالة التشكيل: مُعَلِّم ← معلم
- Unicode NFC

### قواعد canonical:
- استخدم الجذر أو المصدر الصريح
- كل canonical فريدة (لا تكرار)
- aliases = كل الأشكال السطحية الواردة في النص

====================================================
(10) بروتوكول التنفيذ التكراري
====================================================

### الخطوة 1 - اختيار الجذر:
Choose primary concept that is the root
Output:
MindMap
{JSON للجذر فقط مع "children": []}
END_THOUGHT

### الخطوة 2 - فحص إمكانية الإضافة:
Can we add branches?
□ هل توجد معلومات في النص لم تُعالج بعد؟
□ هل يمكن تفصيل عقدة موجودة؟
Output: Yes/No
END_THOUGHT

### الخطوة 3 - إضافة الفروع (إذا Yes):
Add branches:
MindMap
{JSON المحدّث}
END_THOUGHT

**كرر الخطوتين 2 و 3 حتى تصل إلى Output: No**

====================================================
(11) قائمة التحقق (عند كل تكرار)
====================================================

□ هل بقيت معلومات في النص لم تُدرج؟
□ هل كل عقدة مدعومة بالنص فعلاً؟
□ هل يوجد تكرار أو ترادف بين العقد؟
□ هل التسلسل الهرمي منطقي؟
□ هل العلاقات صحيحة؟
□ هل جميع الشروحات مكتوبة بالعربية فقط؟

====================================================
(12) المهمة
====================================================

النص:
{{input_text}}

ابدأ الآن من الخطوة 1 وأكمل جميع التكرارات تلقائياً حتى الوصول إلى Output: No.

⚠️ تذكير نهائي: اكتب جميع الشروحات والتعليقات بالعربية فقط. ممنوع استخدام أي لغة أخرى.'''


# =============================================================================


# Equivalence QA
EQUIVALENCE_QA_PROMPT = """
Check if the following two answers are equivalent.
Use the following format.
Question: question text
Answer 1: answer text
Answer 2: answer text
Conclusion: Yes/No"""


# Global Structure Critic
GLOBAL_STRUCTURE_CRITIC_PROMPT = """Convert paths to Table of Contents (TOC) and evaluate usefulness.

RULES:
- Useful TOC: titles are informative, paths create sensible sentences
- Not Useful TOC: titles are too generic (Birth, Family, Date)

EXAMPLE 1:
Paths:
Assyria -> Assyrian cities -> Aššur
Assyria -> Assyrian cities -> Nineveh
Assyria -> Assyrian language -> Syriac language

TOC:
root((Assyria))
  1. Assyrian cities
    1.1. Aššur
    1.2. Nineveh
  2. Assyrian language
    2.1. Syriac language

Thought: All paths create sensible sentences like "Aššur is an Assyrian city".
Useful: yes

EXAMPLE 2:
Paths:
Lonnie Johnson -> Early life -> Birth
Lonnie Johnson -> Death -> Date
Lonnie Johnson -> Death -> Cause

TOC:
root((Lonnie Johnson))
  1. Early life
    1.1. Birth
  2. Death
    2.1. Date
    2.2. Cause

Thought: "Early life -> Birth" and "Death -> Date" are too generic.
Useful: no

INPUT:
Paths:
{paths}

TOC:

Thought:

Useful:"""


# =============================================================================
# JSON ANALYZER AND CLASSIFIER PROMPT
# =============================================================================

LOCAL_CRITIC_PROMPT = """You are a JSON analyzer and word classifier. Your task is to:
1. Extract ALL leaf node values from the given JSON structure
2. Classify if ALL extracted values are specific content words

══════════════════════════════════════════════════════════════════
STEP 1: EXTRACT LEAF NODES
══════════════════════════════════════════════════════════════════

Leaf nodes are the final values in a JSON structure (not keys).
- Skip all keys/labels
- Extract only the actual values (strings, numbers, dates)
- If a value is nested, go deeper until you reach the final value

══════════════════════════════════════════════════════════════════
STEP 2: CLASSIFY EXTRACTED VALUES
══════════════════════════════════════════════════════════════════

Specific Values (GOOD):
- Names: people, places, organizations (e.g., "Baghdad", "Al-Khwarizmi")
- Numbers and dates (e.g., "1919", "26 May")
- Job titles (e.g., "Lawyer", "Advertising Executive")
- Specific items (e.g., "Astrolabe", "Algebra")

Abstract Concepts (BAD):
- Generic labels (e.g., "Birth", "Family", "Achievements")
- Vague words (e.g., "Things", "Items", "Related")

══════════════════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════════════════

- Answer "yes" if ALL leaf values are specific
- Answer "no" if ANY leaf value is abstract concept

══════════════════════════════════════════════════════════════════
EXAMPLES
══════════════════════════════════════════════════════════════════

JSON:
{{
  "Kay Daly": {{
    "Birth Place": {{
      "City": "Castlecaufield",
      "Country": "Ireland"
    }},
    "Birth Year": "1919",
    "Occupation": "Advertising Executive"
  }}
}}

Extracted Leaf Nodes:
- Castlecaufield
- Ireland
- 1919
- Advertising Executive

Thought: All extracted values are specific - locations, date, and job title.
Answer: yes

---

JSON:
{{
  "Al-Khwarizmi": {{
    "Info": {{
      "Role": "Mathematician",
      "Location": "Baghdad"
    }},
    "Contributions": ["Algebra", "Achievements", "Legacy"]
  }}
}}

Extracted Leaf Nodes:
- Mathematician
- Baghdad
- Algebra
- Achievements
- Legacy

Thought: While "Mathematician", "Baghdad", and "Algebra" are specific,
"Achievements" and "Legacy" are abstract concepts without specific values.
Answer: no

══════════════════════════════════════════════════════════════════
INPUT
══════════════════════════════════════════════════════════════════

JSON:
{json_structure}

Extracted Leaf Nodes:

Thought:

Answer:"""


# Local Critic (Word Classification)
LOCAL_STRUCTURE_CRITIC_PROMPT = """You are a JSON analyzer and word classifier. Your task is to:
1. Extract ALL leaf node values from the given JSON structure
2. Classify if ALL extracted values are specific content words

══════════════════════════════════════════════════════════════════
STEP 1: EXTRACT LEAF NODES
══════════════════════════════════════════════════════════════════

Leaf nodes are the final values in a JSON structure (not keys).
- Skip all keys/labels
- Extract only the actual values (strings, numbers, dates)
- If a value is nested, go deeper until you reach the final value

══════════════════════════════════════════════════════════════════
STEP 2: CLASSIFY EXTRACTED VALUES
══════════════════════════════════════════════════════════════════

Specific Values (GOOD):
- Names: people, places, organizations (e.g., "Baghdad", "Al-Khwarizmi")
- Numbers and dates (e.g., "1919", "26 May")
- Job titles (e.g., "Lawyer", "Advertising Executive")
- Specific items (e.g., "Astrolabe", "Algebra")

Abstract Concepts (BAD):
- Generic labels (e.g., "Birth", "Family", "Achievements")
- Vague words (e.g., "Things", "Items", "Related")

══════════════════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════════════════

- Answer "yes" if ALL leaf values are specific
- Answer "no" if ANY leaf value is abstract concept

══════════════════════════════════════════════════════════════════
EXAMPLES
══════════════════════════════════════════════════════════════════

JSON:
{{
  "Kay Daly": {{
    "Birth Place": {{
      "City": "Castlecaufield",
      "Country": "Ireland"
    }},
    "Birth Year": "1919",
    "Occupation": "Advertising Executive"
  }}
}}

Extracted Leaf Nodes:
- Castlecaufield
- Ireland
- 1919
- Advertising Executive

Thought: All extracted values are specific - locations, date, and job title.
Answer: yes

---

JSON:
{{
  "Al-Khwarizmi": {{
    "Info": {{
      "Role": "Mathematician",
      "Location": "Baghdad"
    }},
    "Contributions": ["Algebra", "Achievements", "Legacy"]
  }}
}}

Extracted Leaf Nodes:
- Mathematician
- Baghdad
- Algebra
- Achievements
- Legacy

Thought: While "Mathematician", "Baghdad", and "Algebra" are specific,
"Achievements" and "Legacy" are abstract concepts without specific values.
Answer: no

══════════════════════════════════════════════════════════════════
INPUT
══════════════════════════════════════════════════════════════════

JSON:
{json_structure}

Extracted Leaf Nodes:

Thought:

Answer:"""


# Bullet Points
BULLET_POINTS_PROMPT = """You are an ULTRA-STRICT text summarizer. MAXIMUM 15 bullet points.

══════════════════════════════════════════════════════════════════
CORE RULE: MERGE ALL RELATED INFORMATION INTO ONE BULLET
══════════════════════════════════════════════════════════════════

If multiple sentences share the SAME subject, merge them into ONE bullet:
- Same location → 1 bullet (include ALL institutions + ALL people)
- Same person → 1 bullet (include ALL achievements)
- Same category → 1 bullet (include ALL items)
- Same cause/effect type → 1 bullet

══════════════════════════════════════════════════════════════════
DETECTION PATTERN
══════════════════════════════════════════════════════════════════

When you see:
- "[Location] had..." + "[Person] worked in [Location]..." → MERGE
- "[Person] did X" + "[Person] also did Y" → MERGE
- "[Category] includes A" + "[Category] includes B" → MERGE
- "Cause 1..." + "Cause 2..." + "Cause 3..." → MERGE into one bullet

══════════════════════════════════════════════════════════════════
FORBIDDEN
══════════════════════════════════════════════════════════════════

❌ Location in one bullet + related people in separate bullets
❌ Starting any bullet with "and", "also", "و", "وكذلك", "ve", "ayrıca"
❌ Exceeding 15 bullet points
❌ Translating any text

══════════════════════════════════════════════════════════════════
OUTPUT RULES
══════════════════════════════════════════════════════════════════

- Each bullet MUST start with a subject (noun), not a connector
- Each bullet MUST be a complete independent sentence
- Keep ALL text in ORIGINAL language
- Number bullets 1-15

Input text:
{input_text}

Bullet points (MAXIMUM 15):"""


# Paths Extraction
PATHS_EXTRACTION_PROMPT = """You are a path extraction expert.

Task: Extract ALL hierarchical paths from the JSON mind map structure.

══════════════════════════════════════════════════════════════════
CRITICAL RULE: REMOVE ALL ARRAY INDICES
══════════════════════════════════════════════════════════════════

When you encounter array index numbers (0, 1, 2, 3...),
you MUST SKIP them completely. Connect parent directly to child value.

DETECTION: If you see a number between " -> " that represents
an array position, DELETE IT from the path.

══════════════════════════════════════════════════════════════════
WRONG vs CORRECT EXAMPLES
══════════════════════════════════════════════════════════════════

❌ WRONG: Centers -> 0 -> City -> Baghdad
✅ CORRECT: Centers -> City -> Baghdad

❌ WRONG: Centers -> 0 -> City -> Baghdad -> Institutions -> 0 -> Library
✅ CORRECT: Centers -> Baghdad -> Institutions -> Library

❌ WRONG: Scientists -> 0 -> Name -> Al-Khwarizmi
✅ CORRECT: Scientists -> Name -> Al-Khwarizmi

❌ WRONG: Scientists -> 0 -> Achievements -> 0 -> Algebra
✅ CORRECT: Scientists -> Achievements -> Algebra

❌ WRONG: Causes -> 0 -> War
✅ CORRECT: Causes -> War

❌ WRONG: Inventions -> Technical -> 0 -> Name -> Astrolabe
✅ CORRECT: Inventions -> Technical -> Name -> Astrolabe

══════════════════════════════════════════════════════════════════
PATH FORMAT RULES
══════════════════════════════════════════════════════════════════

- Use " -> " as separator
- Start each path from root topic
- One path per leaf node
- Plain text output only
- NO JSON format
- NO array indices (0, 1, 2, 3...)
- Keep ORIGINAL language
- Do NOT translate

══════════════════════════════════════════════════════════════════
POST-PROCESSING CHECK
══════════════════════════════════════════════════════════════════

Before outputting, scan each path:
1. Find any standalone numbers between " -> "
2. If number is array index (0, 1, 2...), REMOVE IT
3. Reconnect the path without the number

Example fix:
"A -> 0 -> B -> 1 -> C" → Remove 0 and 1 → "A -> B -> C"

══════════════════════════════════════════════════════════════════
ABSOLUTELY FORBIDDEN
══════════════════════════════════════════════════════════════════

❌ Any path containing " -> 0 -> "
❌ Any path containing " -> 1 -> "
❌ Any path containing " -> 2 -> "
❌ Any standalone number as path segment
❌ JSON format output
❌ Translation

══════════════════════════════════════════════════════════════════

JSON:
{json_structure}

Paths (NO INDEX NUMBERS):"""


# Factual Critic
FACTUAL_CRITIC_PROMPT = """You are a factual attribution expert.

Task: Link each path to its source sentence from the input text.

Rules:
- For each path, find the sentence that supports it
- Use format: path [sentence_number]
- Use [NA] if no supporting sentence found
- Keep all text in original language (Arabic, Turkish, English, etc.)
- Do NOT translate anything
- Only cite if clearly supported by the text

Example:
Input text:
1. The capital of France is Paris.
2. It has a population of 2 million people.
3. The Eiffel Tower is located there.

Paths:
France -> capital -> Paris
France -> population -> 5 million
Paris -> landmark -> Eiffel Tower

Paths with attribution:
France -> capital -> Paris [1]
France -> population -> 5 million [NA]
Paris -> landmark -> Eiffel Tower [3]

Now process:

Input text:
{bullet_points}

Paths:
{paths}

Paths with attribution:"""


# Factual Validator
FACTUAL_VALIDATOR_PROMPT = """You are a factuality validator. Your task is to check if a structured
output (table or mind map) is factually grounded in the source text.

You will receive a list of paths with their citations in the format:
Path -> Citation: [X] or [NA]

Rules:
- [X] means the information is supported by sentence X in the source text
- [NA] means the information is NOT found in the source text (hallucination)

Your task:
1. Count the total number of paths
2. Count the number of paths with [NA]
3. Make a decision based on Zero Tolerance policy:
   - If ANY path has [NA] → REJECT
   - If ALL paths have valid citations → ACCEPT

Output ONLY valid JSON in this exact format without ```json ```:
{{
  "total_paths": <number>,
  "paths_with_na": <number>,
  "paths_with_valid_citations": <number>,
  "factuality_score": <percentage>,
  "decision": "ACCEPT" or "REJECT",
  "reason": "<explanation>"
}}

---
Paths to evaluate:
{paths_with_citations}"""


# Auto QA
AUTO_QA_PROMPT = """Generate fact-based QA pairs from text.

RULES:
- Use ONLY explicit text information
- NO inference or assumptions
- NO repeated questions
- Answer must exist word-for-word in text

Output format (SINGLE array):
{{"qa_pairs":[
  {{"question":"...","answer":"...","source_sentence":1}},
  {{"question":"...","answer":"...","source_sentence":2}}
]}}

Text:
{text}"""


# Mermaid
MERMAID_PROMPT = """You are a Mermaid diagram expert. Convert the JSON below into a Mermaid mindmap.

Rules:
- Start with "mindmap" keyword
- Use 2-space indentation for hierarchy
- Keep ALL text in its original language (Arabic, Turkish, English, etc.)
- Do NOT translate anything
- Remove special characters: () [] {{}} < > # & " '
- Output ONLY Mermaid code, no explanations
- Mustn't write ``` mermaid ```

JSON:
{json_structure}

Mermaid mindmap:"""


# JSON Repair
JSON_REPAIR_PROMPT = """You are a JSON repair expert. Fix the broken JSON below.

Rules:
- Fix all syntax errors (missing quotes, commas, brackets, colons)
- Preserve ALL original text in its original language (Arabic, Turkish, English, etc.)
- Do NOT translate or modify any text content
- Output ONLY valid JSON, no explanations

Broken JSON:
{broken_json}

Fixed JSON:"""


# QA Validity
QA_VALIDITY_PROMPT = """Answer in concise manner the question using the information below.
Say <unknown> when the questions cannot be answered.

{data}

Question:
{question}"""


# =============================================================================
# PUBLIC API
# =============================================================================

# Mind-map system prompts, keyed by language code.
MINDMAP_PROMPTS = {
    "en": ENGLISH_PROMPT,
    "tr": TURKISH_PROMPT,
    "ar": ARABIC_PROMPT,
}

LANGUAGE_NAMES = {"en": "English", "tr": "Turkish", "ar": "Arabic"}

# Placeholder tokens for the input text inside the mind-map system prompts.
# (The notebooks use slightly different tokens per language; we normalise here.)
_INPUT_PLACEHOLDERS = ["{{ input_text }}", "{{input_text}}", "{input_text}"]


def get_mindmap_system_prompt(language: str, input_text: str | None = None) -> str:
    """Return the mind-map system prompt for a language.

    If ``input_text`` is given, the input-text placeholder is filled in-place so
    the prompt is self-contained and presentation-ready; otherwise the raw
    template (with placeholder) is returned.
    """
    lang = language.lower()
    if lang not in MINDMAP_PROMPTS:
        raise ValueError(f"Unknown language: {lang}. Use one of: en, tr, ar")
    prompt = MINDMAP_PROMPTS[lang]
    if input_text is not None:
        for token in _INPUT_PLACEHOLDERS:
            prompt = prompt.replace(token, input_text)
    return prompt


# Pipeline / critic prompt registry.
PIPELINE_PROMPTS = {
    "json_repair": JSON_REPAIR_PROMPT,
    "paths_extraction": PATHS_EXTRACTION_PROMPT,
    "local_structure_critic": LOCAL_STRUCTURE_CRITIC_PROMPT,
    "global_structure_critic": GLOBAL_STRUCTURE_CRITIC_PROMPT,
    "bullet_points": BULLET_POINTS_PROMPT,
    "factual_critic": FACTUAL_CRITIC_PROMPT,
    "factual_validator": FACTUAL_VALIDATOR_PROMPT,
    "mermaid": MERMAID_PROMPT,
}


def build_pipeline_prompt(name: str, **kwargs) -> str:
    """Return a pipeline prompt with its placeholders filled in.

    Unlike the notebook helper, this does NOT print and simply returns the
    rendered string.
    """
    if name not in PIPELINE_PROMPTS:
        raise ValueError(f"Unknown pipeline prompt: {name}")
    return PIPELINE_PROMPTS[name].format(**kwargs)

==========================================
=== STEP 10 — VREST OVERLAP RECHECK ===
==========================================

##########################################
### FEATURE: vrest
##########################################

--- SPLIT: TRAINING (12 drugs) ---
  bepridil        (Label: 0) | min: -1.0001745660 | max: -1.0000324030 | mean: -1.0000542412 | std: 0.0000180226
  chlorpromazine  (Label: 1) | min: -1.0000673580 | max: -0.9999865040 | mean: -1.0000027493 | std: 0.0000100042
  cisapride       (Label: 1) | min: -1.0000105830 | max: -0.9999851820 | mean: -0.9999904506 | std: 0.0000018281
  diltiazem       (Label: 2) | min: -1.0000914580 | max: -1.0000011440 | mean: -1.0000249853 | std: 0.0000097761
  dofetilide      (Label: 0) | min: -1.0000267430 | max: -1.0000040550 | mean: -1.0000089842 | std: 0.0000023356
  mexiletine      (Label: 2) | min: -1.0004910280 | max: -1.0001537700 | mean: -1.0002470864 | std: 0.0000451562
  ondansetron     (Label: 1) | min: -1.0000577630 | max: -0.9999640160 | mean: -0.9999951329 | std: 0.0000187266
  quinidine       (Label: 0) | min: -1.0005342580 | max: -1.0002251220 | mean: -1.0003001742 | std: 0.0000451876
  ranolazine      (Label: 2) | min: -1.0002255380 | max: -1.0000556260 | mean: -1.0000893520 | std: 0.0000135196
  sotalol         (Label: 0) | min: -1.0000166110 | max: -0.9999579040 | mean: -0.9999696871 | std: 0.0000057125
  terfenadine     (Label: 1) | min: -1.0000562180 | max: -1.0000017880 | mean: -1.0000189336 | std: 0.0000074531
  verapamil       (Label: 2) | min: -1.0003295670 | max: -1.0000579730 | mean: -1.0001774058 | std: 0.0000366206

Total Pairs: 66 | Overlapping Pairs: 40 (60.61%)
Up to 10 overlapping pairs with ranges:
   1. bepridil     [-1.0001745660, -1.0000324030] vs chlorpromazine [-1.0000673580, -0.9999865040]
   2. bepridil     [-1.0001745660, -1.0000324030] vs diltiazem    [-1.0000914580, -1.0000011440]
   3. bepridil     [-1.0001745660, -1.0000324030] vs mexiletine   [-1.0004910280, -1.0001537700]
   4. bepridil     [-1.0001745660, -1.0000324030] vs ondansetron  [-1.0000577630, -0.9999640160]
   5. bepridil     [-1.0001745660, -1.0000324030] vs ranolazine   [-1.0002255380, -1.0000556260]
   6. bepridil     [-1.0001745660, -1.0000324030] vs terfenadine  [-1.0000562180, -1.0000017880]
   7. bepridil     [-1.0001745660, -1.0000324030] vs verapamil    [-1.0003295670, -1.0000579730]
   8. chlorpromazine [-1.0000673580, -0.9999865040] vs cisapride    [-1.0000105830, -0.9999851820]
   9. chlorpromazine [-1.0000673580, -0.9999865040] vs diltiazem    [-1.0000914580, -1.0000011440]
  10. chlorpromazine [-1.0000673580, -0.9999865040] vs dofetilide   [-1.0000267430, -1.0000040550]

--- SPLIT: TESTING (16 drugs) ---
  astemizole      (Label: 1) | min: -1.0000016550 | max: -0.9999967260 | mean: -0.9999975862 | std: 0.0000007647
  azimilide       (Label: 0) | min: -1.0000029420 | max: -0.9999630810 | mean: -0.9999806697 | std: 0.0000053325
  clarithromycin  (Label: 1) | min: -1.0000653580 | max: -0.9999624230 | mean: -0.9999847510 | std: 0.0000140394
  clozapine       (Label: 1) | min: -1.0001316010 | max: -0.9999836280 | mean: -1.0000054305 | std: 0.0000129163
  disopyramide    (Label: 0) | min: -1.0001828770 | max: -0.9999561650 | mean: -1.0000217087 | std: 0.0000249614
  domperidone     (Label: 1) | min: -1.0001121430 | max: -1.0000106430 | mean: -1.0000399327 | std: 0.0000137532
  droperidol      (Label: 1) | min: -1.0000417170 | max: -0.9999788130 | mean: -0.9999969181 | std: 0.0000075624
  ibutilide       (Label: 0) | min: -1.0002616800 | max: -0.9997696490 | mean: -1.0001642514 | std: 0.0000681103
  loratadine      (Label: 2) | min: -1.0000052990 | max: -0.9999991770 | mean: -0.9999999579 | std: 0.0000003146
  metoprolol      (Label: 2) | min: -1.0001394560 | max: -0.9999749720 | mean: -1.0000496957 | std: 0.0000215100
  nifedipine      (Label: 2) | min: -1.0007831470 | max: -1.0000966560 | mean: -1.0003810785 | std: 0.0000789415
  nitrendipine    (Label: 2) | min: -1.0001910260 | max: -1.0000003720 | mean: -1.0000461185 | std: 0.0000337564
  pimozide        (Label: 1) | min: -1.0000560540 | max: -0.9999984900 | mean: -1.0000171872 | std: 0.0000096413
  risperidone     (Label: 1) | min: -1.0000041310 | max: -0.9999903580 | mean: -0.9999962617 | std: 0.0000022245
  tamoxifen       (Label: 2) | min: -1.0000324520 | max: -0.9999910490 | mean: -0.9999976806 | std: 0.0000023268
  vandetanib      (Label: 0) | min: -1.0002314220 | max: -0.9999938960 | mean: -1.0000405963 | std: 0.0000246943

Total Pairs: 120 | Overlapping Pairs: 108 (90.00%)
Up to 10 overlapping pairs with ranges:
   1. astemizole   [-1.0000016550, -0.9999967260] vs azimilide    [-1.0000029420, -0.9999630810]
   2. astemizole   [-1.0000016550, -0.9999967260] vs clarithromycin [-1.0000653580, -0.9999624230]
   3. astemizole   [-1.0000016550, -0.9999967260] vs clozapine    [-1.0001316010, -0.9999836280]
   4. astemizole   [-1.0000016550, -0.9999967260] vs disopyramide [-1.0001828770, -0.9999561650]
   5. astemizole   [-1.0000016550, -0.9999967260] vs droperidol   [-1.0000417170, -0.9999788130]
   6. astemizole   [-1.0000016550, -0.9999967260] vs ibutilide    [-1.0002616800, -0.9997696490]
   7. astemizole   [-1.0000016550, -0.9999967260] vs loratadine   [-1.0000052990, -0.9999991770]
   8. astemizole   [-1.0000016550, -0.9999967260] vs metoprolol   [-1.0001394560, -0.9999749720]
   9. astemizole   [-1.0000016550, -0.9999967260] vs nitrendipine [-1.0001910260, -1.0000003720]
  10. astemizole   [-1.0000016550, -0.9999967260] vs pimozide     [-1.0000560540, -0.9999984900]

##########################################
### FEATURE: qNet
##########################################

--- SPLIT: TRAINING (12 drugs) ---
  bepridil        (Label: 0) | min: 0.6096315240 | max: 0.7831251920 | mean: 0.6706143158 | std: 0.0224822672
  chlorpromazine  (Label: 1) | min: 0.9085297630 | max: 1.0028697900 | mean: 0.9505021265 | std: 0.0120320788
  cisapride       (Label: 1) | min: 0.8309762910 | max: 0.9161244780 | mean: 0.8690398044 | std: 0.0133394077
  diltiazem       (Label: 2) | min: 0.9982802430 | max: 1.1122601830 | mean: 1.0412189068 | std: 0.0131201775
  dofetilide      (Label: 0) | min: 0.6568179640 | max: 0.7708740170 | mean: 0.7180914988 | std: 0.0171076039
  mexiletine      (Label: 2) | min: 1.2561731590 | max: 1.4659063360 | mean: 1.3371207795 | std: 0.0283280932
  ondansetron     (Label: 1) | min: 0.8515377330 | max: 0.9413041160 | mean: 0.8925500043 | std: 0.0148254215
  quinidine       (Label: 0) | min: 0.5693540700 | max: 0.9616465220 | mean: 0.7566648095 | std: 0.0523429466
  ranolazine      (Label: 2) | min: 1.0365226710 | max: 1.1594711380 | mean: 1.1004663544 | std: 0.0183672396
  sotalol         (Label: 0) | min: 0.7461832180 | max: 0.8583515040 | mean: 0.7988036859 | std: 0.0173028560
  terfenadine     (Label: 1) | min: 0.8133614560 | max: 1.0104102950 | mean: 0.8883996276 | std: 0.0277004739
  verapamil       (Label: 2) | min: 0.8800138410 | max: 1.1118126380 | mean: 1.0161633889 | std: 0.0324230289

Total Pairs: 66 | Overlapping Pairs: 29 (43.94%)
Up to 10 overlapping pairs with ranges:
   1. bepridil     [0.6096315240, 0.7831251920] vs dofetilide   [0.6568179640, 0.7708740170]
   2. bepridil     [0.6096315240, 0.7831251920] vs quinidine    [0.5693540700, 0.9616465220]
   3. bepridil     [0.6096315240, 0.7831251920] vs sotalol      [0.7461832180, 0.8583515040]
   4. chlorpromazine [0.9085297630, 1.0028697900] vs cisapride    [0.8309762910, 0.9161244780]
   5. chlorpromazine [0.9085297630, 1.0028697900] vs diltiazem    [0.9982802430, 1.1122601830]
   6. chlorpromazine [0.9085297630, 1.0028697900] vs ondansetron  [0.8515377330, 0.9413041160]
   7. chlorpromazine [0.9085297630, 1.0028697900] vs quinidine    [0.5693540700, 0.9616465220]
   8. chlorpromazine [0.9085297630, 1.0028697900] vs terfenadine  [0.8133614560, 1.0104102950]
   9. chlorpromazine [0.9085297630, 1.0028697900] vs verapamil    [0.8800138410, 1.1118126380]
  10. cisapride    [0.8309762910, 0.9161244780] vs ondansetron  [0.8515377330, 0.9413041160]

--- SPLIT: TESTING (16 drugs) ---
  astemizole      (Label: 1) | min: 0.7772632630 | max: 0.9547400330 | mean: 0.8660535490 | std: 0.0272202148
  azimilide       (Label: 0) | min: 0.5808691230 | max: 0.7580786320 | mean: 0.6609536237 | std: 0.0295743780
  clarithromycin  (Label: 1) | min: 0.8750666390 | max: 0.9793861530 | mean: 0.9249794368 | std: 0.0164045486
  clozapine       (Label: 1) | min: 0.8944353300 | max: 1.0514849260 | mean: 0.9561415166 | std: 0.0225074511
  disopyramide    (Label: 0) | min: 0.7328537540 | max: 0.9732149530 | mean: 0.8617794936 | std: 0.0324098425
  domperidone     (Label: 1) | min: 0.5681612750 | max: 0.8408714840 | mean: 0.6706412968 | std: 0.0380878437
  droperidol      (Label: 1) | min: 0.7836542320 | max: 0.9590322070 | mean: 0.8755407494 | std: 0.0286385917
  ibutilide       (Label: 0) | min: -0.2802466520 | max: 0.5503893070 | mean: 0.2743824947 | std: 0.1527652674
  loratadine      (Label: 2) | min: 0.9959531110 | max: 1.0113006210 | mean: 0.9995841926 | std: 0.0006632038
  metoprolol      (Label: 2) | min: 0.8825638990 | max: 1.1541356090 | mean: 1.0364518347 | std: 0.0374476633
  nifedipine      (Label: 2) | min: 1.0595564230 | max: 1.3518513030 | mean: 1.1908667658 | std: 0.0359861671
  nitrendipine    (Label: 2) | min: 0.9999226080 | max: 1.1053655790 | mean: 1.0365180449 | std: 0.0169279753
  pimozide        (Label: 1) | min: 0.5175594820 | max: 0.8382090730 | mean: 0.6656821574 | std: 0.0701318736
  risperidone     (Label: 1) | min: 0.9004131410 | max: 0.9987961790 | mean: 0.9621476077 | std: 0.0204122400
  tamoxifen       (Label: 2) | min: 0.9355629510 | max: 1.0023036620 | mean: 0.9818201775 | std: 0.0099739731
  vandetanib      (Label: 0) | min: 0.4114876330 | max: 0.7591112220 | mean: 0.5219537250 | std: 0.0381225571

Total Pairs: 120 | Overlapping Pairs: 54 (45.00%)
Up to 10 overlapping pairs with ranges:
   1. astemizole   [0.7772632630, 0.9547400330] vs clarithromycin [0.8750666390, 0.9793861530]
   2. astemizole   [0.7772632630, 0.9547400330] vs clozapine    [0.8944353300, 1.0514849260]
   3. astemizole   [0.7772632630, 0.9547400330] vs disopyramide [0.7328537540, 0.9732149530]
   4. astemizole   [0.7772632630, 0.9547400330] vs domperidone  [0.5681612750, 0.8408714840]
   5. astemizole   [0.7772632630, 0.9547400330] vs droperidol   [0.7836542320, 0.9590322070]
   6. astemizole   [0.7772632630, 0.9547400330] vs metoprolol   [0.8825638990, 1.1541356090]
   7. astemizole   [0.7772632630, 0.9547400330] vs pimozide     [0.5175594820, 0.8382090730]
   8. astemizole   [0.7772632630, 0.9547400330] vs risperidone  [0.9004131410, 0.9987961790]
   9. astemizole   [0.7772632630, 0.9547400330] vs tamoxifen    [0.9355629510, 1.0023036620]
  10. azimilide    [0.5808691230, 0.7580786320] vs disopyramide [0.7328537540, 0.9732149530]

##########################################
### FEATURE: carest
##########################################

--- SPLIT: TRAINING (12 drugs) ---
  bepridil        (Label: 0) | min: 0.9830000000 | max: 1.0200000000 | mean: 1.0090295000 | std: 0.0033171720
  chlorpromazine  (Label: 1) | min: 0.9890000000 | max: 1.0100000000 | mean: 1.0020935000 | std: 0.0042427300
  cisapride       (Label: 1) | min: 1.0000000000 | max: 1.0100000000 | mean: 1.0099800000 | std: 0.0004467662
  diltiazem       (Label: 2) | min: 0.9830000000 | max: 1.0000000000 | mean: 0.9989280000 | std: 0.0018129578
  dofetilide      (Label: 0) | min: 1.0100000000 | max: 1.0100000000 | mean: 1.0100000000 | std: 0.0000000000
  mexiletine      (Label: 2) | min: 0.9450000000 | max: 1.0100000000 | mean: 0.9909600000 | std: 0.0086100755
  ondansetron     (Label: 1) | min: 0.9920000000 | max: 1.0100000000 | mean: 1.0051800000 | std: 0.0052784089
  quinidine       (Label: 0) | min: 0.9630000000 | max: 1.0200000000 | mean: 1.0064745000 | std: 0.0079945825
  ranolazine      (Label: 2) | min: 0.9820000000 | max: 1.0200000000 | mean: 1.0102950000 | std: 0.0034679929
  sotalol         (Label: 0) | min: 1.0000000000 | max: 1.0200000000 | mean: 1.0101600000 | std: 0.0014051334
  terfenadine     (Label: 1) | min: 1.0000000000 | max: 1.0100000000 | mean: 1.0094800000 | std: 0.0022202703
  verapamil       (Label: 2) | min: 0.9390000000 | max: 0.9980000000 | mean: 0.9724275000 | std: 0.0080647222

Total Pairs: 66 | Overlapping Pairs: 61 (92.42%)
Up to 10 overlapping pairs with ranges:
   1. bepridil     [0.9830000000, 1.0200000000] vs chlorpromazine [0.9890000000, 1.0100000000]
   2. bepridil     [0.9830000000, 1.0200000000] vs cisapride    [1.0000000000, 1.0100000000]
   3. bepridil     [0.9830000000, 1.0200000000] vs diltiazem    [0.9830000000, 1.0000000000]
   4. bepridil     [0.9830000000, 1.0200000000] vs dofetilide   [1.0100000000, 1.0100000000]
   5. bepridil     [0.9830000000, 1.0200000000] vs mexiletine   [0.9450000000, 1.0100000000]
   6. bepridil     [0.9830000000, 1.0200000000] vs ondansetron  [0.9920000000, 1.0100000000]
   7. bepridil     [0.9830000000, 1.0200000000] vs quinidine    [0.9630000000, 1.0200000000]
   8. bepridil     [0.9830000000, 1.0200000000] vs ranolazine   [0.9820000000, 1.0200000000]
   9. bepridil     [0.9830000000, 1.0200000000] vs sotalol      [1.0000000000, 1.0200000000]
  10. bepridil     [0.9830000000, 1.0200000000] vs terfenadine  [1.0000000000, 1.0100000000]

--- SPLIT: TESTING (16 drugs) ---
  astemizole      (Label: 1) | min: 1.0000000000 | max: 1.0100000000 | mean: 1.0092250000 | std: 0.0026738315
  azimilide       (Label: 0) | min: 1.0100000000 | max: 1.0200000000 | mean: 1.0198700000 | std: 0.0011327400
  clarithromycin  (Label: 1) | min: 0.9880000000 | max: 1.0100000000 | mean: 1.0063425000 | std: 0.0051248604
  clozapine       (Label: 1) | min: 0.9750000000 | max: 1.0100000000 | mean: 1.0009155000 | std: 0.0038335831
  disopyramide    (Label: 0) | min: 0.9750000000 | max: 1.0200000000 | mean: 1.0108180000 | std: 0.0067597985
  domperidone     (Label: 1) | min: 1.0000000000 | max: 1.0200000000 | mean: 1.0100550000 | std: 0.0010232180
  droperidol      (Label: 1) | min: 1.0000000000 | max: 1.0100000000 | mean: 1.0094400000 | std: 0.0022992173
  ibutilide       (Label: 0) | min: 0.9590000000 | max: 1.0200000000 | mean: 0.9903630000 | std: 0.0083813621
  loratadine      (Label: 2) | min: 1.0000000000 | max: 1.0000000000 | mean: 1.0000000000 | std: 0.0000000000
  metoprolol      (Label: 2) | min: 0.9850000000 | max: 1.0100000000 | mean: 1.0029795000 | std: 0.0053062303
  nifedipine      (Label: 2) | min: 0.8360000000 | max: 0.9820000000 | mean: 0.9167540000 | std: 0.0173187033
  nitrendipine    (Label: 2) | min: 0.9580000000 | max: 1.0000000000 | mean: 0.9911855000 | std: 0.0077601604
  pimozide        (Label: 1) | min: 1.0000000000 | max: 1.0100000000 | mean: 1.0099700000 | std: 0.0005469004
  risperidone     (Label: 1) | min: 1.0000000000 | max: 1.0100000000 | mean: 1.0000350000 | std: 0.0005905718
  tamoxifen       (Label: 2) | min: 0.9930000000 | max: 1.0000000000 | mean: 0.9999840000 | std: 0.0002659774
  vandetanib      (Label: 0) | min: 0.9620000000 | max: 1.0200000000 | mean: 1.0041270000 | std: 0.0061768010

Total Pairs: 120 | Overlapping Pairs: 107 (89.17%)
Up to 10 overlapping pairs with ranges:
   1. astemizole   [1.0000000000, 1.0100000000] vs azimilide    [1.0100000000, 1.0200000000]
   2. astemizole   [1.0000000000, 1.0100000000] vs clarithromycin [0.9880000000, 1.0100000000]
   3. astemizole   [1.0000000000, 1.0100000000] vs clozapine    [0.9750000000, 1.0100000000]
   4. astemizole   [1.0000000000, 1.0100000000] vs disopyramide [0.9750000000, 1.0200000000]
   5. astemizole   [1.0000000000, 1.0100000000] vs domperidone  [1.0000000000, 1.0200000000]
   6. astemizole   [1.0000000000, 1.0100000000] vs droperidol   [1.0000000000, 1.0100000000]
   7. astemizole   [1.0000000000, 1.0100000000] vs ibutilide    [0.9590000000, 1.0200000000]
   8. astemizole   [1.0000000000, 1.0100000000] vs loratadine   [1.0000000000, 1.0000000000]
   9. astemizole   [1.0000000000, 1.0100000000] vs metoprolol   [0.9850000000, 1.0100000000]
  10. astemizole   [1.0000000000, 1.0100000000] vs nitrendipine [0.9580000000, 1.0000000000]

##########################################
### FEATURE: dvdtmax
##########################################

--- SPLIT: TRAINING (12 drugs) ---
  bepridil        (Label: 0) | min: 0.9935623770 | max: 1.0013315460 | mean: 1.0009686226 | std: 0.0004574083
  chlorpromazine  (Label: 1) | min: 0.9932937980 | max: 1.0001696330 | mean: 0.9995180547 | std: 0.0007287607
  cisapride       (Label: 1) | min: 0.9982584110 | max: 1.0002116680 | mean: 0.9998471808 | std: 0.0003248670
  diltiazem       (Label: 2) | min: 0.9980434620 | max: 1.0000421890 | mean: 0.9994369183 | std: 0.0002654195
  dofetilide      (Label: 0) | min: 0.9986917100 | max: 1.0007507230 | mean: 1.0005542532 | std: 0.0001127626
  mexiletine      (Label: 2) | min: 0.9626594150 | max: 0.9979544540 | mean: 0.9866813586 | std: 0.0043770268
  ondansetron     (Label: 1) | min: 0.9935764670 | max: 0.9998979990 | mean: 0.9993079587 | std: 0.0006824093
  quinidine       (Label: 0) | min: 0.9005086980 | max: 0.9690423460 | mean: 0.9382742542 | std: 0.0091166728
  ranolazine      (Label: 2) | min: 0.9761469780 | max: 1.0003633280 | mean: 0.9932095250 | std: 0.0038118230
  sotalol         (Label: 0) | min: 0.9955389960 | max: 0.9999836120 | mean: 0.9994242181 | std: 0.0005982632
  terfenadine     (Label: 1) | min: 0.9992343170 | max: 1.0008249450 | mean: 1.0004049105 | std: 0.0001207291
  verapamil       (Label: 2) | min: 0.9937249420 | max: 1.0010068490 | mean: 1.0003045865 | std: 0.0006333778

Total Pairs: 66 | Overlapping Pairs: 52 (78.79%)
Up to 10 overlapping pairs with ranges:
   1. bepridil     [0.9935623770, 1.0013315460] vs chlorpromazine [0.9932937980, 1.0001696330]
   2. bepridil     [0.9935623770, 1.0013315460] vs cisapride    [0.9982584110, 1.0002116680]
   3. bepridil     [0.9935623770, 1.0013315460] vs diltiazem    [0.9980434620, 1.0000421890]
   4. bepridil     [0.9935623770, 1.0013315460] vs dofetilide   [0.9986917100, 1.0007507230]
   5. bepridil     [0.9935623770, 1.0013315460] vs mexiletine   [0.9626594150, 0.9979544540]
   6. bepridil     [0.9935623770, 1.0013315460] vs ondansetron  [0.9935764670, 0.9998979990]
   7. bepridil     [0.9935623770, 1.0013315460] vs ranolazine   [0.9761469780, 1.0003633280]
   8. bepridil     [0.9935623770, 1.0013315460] vs sotalol      [0.9955389960, 0.9999836120]
   9. bepridil     [0.9935623770, 1.0013315460] vs terfenadine  [0.9992343170, 1.0008249450]
  10. bepridil     [0.9935623770, 1.0013315460] vs verapamil    [0.9937249420, 1.0010068490]

--- SPLIT: TESTING (16 drugs) ---
  astemizole      (Label: 1) | min: 1.0000497100 | max: 1.0004084840 | mean: 1.0002065775 | std: 0.0000556109
  azimilide       (Label: 0) | min: 0.9964499560 | max: 1.0003589770 | mean: 1.0000068843 | std: 0.0002418794
  clarithromycin  (Label: 1) | min: 0.9924531090 | max: 0.9998981520 | mean: 0.9993996949 | std: 0.0006441624
  clozapine       (Label: 1) | min: 0.9979598560 | max: 1.0002049260 | mean: 0.9998006192 | std: 0.0002001891
  disopyramide    (Label: 0) | min: 0.9711195130 | max: 1.0004135980 | mean: 0.9952832408 | std: 0.0043326123
  domperidone     (Label: 1) | min: 0.9925940560 | max: 1.0015557860 | mean: 1.0006849098 | std: 0.0006576666
  droperidol      (Label: 1) | min: 0.9967075580 | max: 1.0003928950 | mean: 0.9998708967 | std: 0.0002955579
  ibutilide       (Label: 0) | min: 0.9530057670 | max: 1.0026148000 | mean: 0.9913893648 | std: 0.0080814921
  loratadine      (Label: 2) | min: 0.9993574350 | max: 1.0000434940 | mean: 0.9999962343 | std: 0.0000244121
  metoprolol      (Label: 2) | min: 0.9845484080 | max: 1.0001736850 | mean: 0.9960306902 | std: 0.0021893902
  nifedipine      (Label: 2) | min: 0.9978621750 | max: 1.0030992940 | mean: 1.0009093801 | std: 0.0003375301
  nitrendipine    (Label: 2) | min: 0.9962963720 | max: 1.0004018270 | mean: 1.0000552384 | std: 0.0002481052
  pimozide        (Label: 1) | min: 1.0002890400 | max: 1.0011630630 | mean: 1.0007241833 | std: 0.0001975939
  risperidone     (Label: 1) | min: 0.9997469010 | max: 1.0001244090 | mean: 1.0000130643 | std: 0.0000241082
  tamoxifen       (Label: 2) | min: 0.9984470980 | max: 1.0000229520 | mean: 0.9999724362 | std: 0.0000460927
  vandetanib      (Label: 0) | min: 0.9747418080 | max: 0.9997937900 | mean: 0.9977084919 | std: 0.0015516641

Total Pairs: 120 | Overlapping Pairs: 109 (90.83%)
Up to 10 overlapping pairs with ranges:
   1. astemizole   [1.0000497100, 1.0004084840] vs azimilide    [0.9964499560, 1.0003589770]
   2. astemizole   [1.0000497100, 1.0004084840] vs clozapine    [0.9979598560, 1.0002049260]
   3. astemizole   [1.0000497100, 1.0004084840] vs disopyramide [0.9711195130, 1.0004135980]
   4. astemizole   [1.0000497100, 1.0004084840] vs domperidone  [0.9925940560, 1.0015557860]
   5. astemizole   [1.0000497100, 1.0004084840] vs droperidol   [0.9967075580, 1.0003928950]
   6. astemizole   [1.0000497100, 1.0004084840] vs ibutilide    [0.9530057670, 1.0026148000]
   7. astemizole   [1.0000497100, 1.0004084840] vs metoprolol   [0.9845484080, 1.0001736850]
   8. astemizole   [1.0000497100, 1.0004084840] vs nifedipine   [0.9978621750, 1.0030992940]
   9. astemizole   [1.0000497100, 1.0004084840] vs nitrendipine [0.9962963720, 1.0004018270]
  10. astemizole   [1.0000497100, 1.0004084840] vs pimozide     [1.0002890400, 1.0011630630]

==========================================
=== SUMMARY TABLE: FEATURE x SPLIT ===
==========================================
Feature    | Split      | Overlap Count   | Total Pairs  | Percentage
-----------------------------------------------------------------
vrest      | TRAINING   |              40 |           66 |  60.61%
vrest      | TESTING    |             108 |          120 |  90.00%
qNet       | TRAINING   |              29 |           66 |  43.94%
qNet       | TESTING    |              54 |          120 |  45.00%
carest     | TRAINING   |              61 |           66 |  92.42%
carest     | TESTING    |             107 |          120 |  89.17%
dvdtmax    | TRAINING   |              52 |           66 |  78.79%
dvdtmax    | TESTING    |             109 |          120 |  90.83%

--- VERDICT ON READINGS ---
Recomputed vrest overlaps: Training = 40, Testing = 108.
The recomputed vrest numbers support the reading of 40 (train) and 108 (test) overlapping pairs.

==========================================
=== GREP docs/STEP8_SHAP_DIAGNOSTIC.md FOR OVERLAP SECTION ===
==========================================
--- Analysis for vrest ---

TRAINING (12 drugs):
  bepridil        (True: 0) | min: -1.00017457 | max: -1.00003240 | mean: -1.00005424 | std: 0.00001803
  chlorpromazine  (True: 1) | min: -1.00006736 | max: -0.99998650 | mean: -1.00000275 | std: 0.00001001
  cisapride       (True: 1) | min: -1.00001058 | max: -0.99998518 | mean: -0.99999045 | std: 0.00000183
  diltiazem       (True: 2) | min: -1.00009146 | max: -1.00000114 | mean: -1.00002499 | std: 0.00000978
  dofetilide      (True: 0) | min: -1.00002674 | max: -1.00000406 | mean: -1.00000898 | std: 0.00000234
  mexiletine      (True: 2) | min: -1.00049103 | max: -1.00015377 | mean: -1.00024709 | std: 0.00004517
  ondansetron     (True: 1) | min: -1.00005776 | max: -0.99996402 | mean: -0.99999513 | std: 0.00001873
  quinidine       (True: 0) | min: -1.00053426 | max: -1.00022512 | mean: -1.00030017 | std: 0.00004520
  ranolazine      (True: 2) | min: -1.00022554 | max: -1.00005563 | mean: -1.00008935 | std: 0.00001352
  sotalol         (True: 0) | min: -1.00001661 | max: -0.99995790 | mean: -0.99996969 | std: 0.00000571
  terfenadine     (True: 1) | min: -1.00005622 | max: -1.00000179 | mean: -1.00001893 | std: 0.00000745
  verapamil       (True: 2) | min: -1.00032957 | max: -1.00005797 | mean: -1.00017741 | std: 0.00003663
  Overlapping pairs: 40
  Up to 10 examples: [('quinidine', 'mexiletine'), ('quinidine', 'verapamil'), ('quinidine', 'ranolazine'), ('mexiletine', 'verapamil'), ('mexiletine', 'ranolazine'), ('mexiletine', 'bepridil'), ('verapamil', 'ranolazine'), ('verapamil', 'bepridil'), ('verapamil', 'diltiazem'), ('verapamil', 'chlorpromazine')]
  Can the drugs be perfectly ordered/separated by vrest alone? NO, ranges overlap.

TESTING (16 drugs):
  astemizole      (True: 1) | min: -1.00000165 | max: -0.99999673 | mean: -0.99999759 | std: 0.00000076
  azimilide       (True: 0) | min: -1.00000294 | max: -0.99996308 | mean: -0.99998067 | std: 0.00000533
  clarithromycin  (True: 1) | min: -1.00006536 | max: -0.99996242 | mean: -0.99998475 | std: 0.00001404
  clozapine       (True: 1) | min: -1.00013160 | max: -0.99998363 | mean: -1.00000543 | std: 0.00001292
  disopyramide    (True: 0) | min: -1.00018288 | max: -0.99995616 | mean: -1.00002171 | std: 0.00002497
  domperidone     (True: 1) | min: -1.00011214 | max: -1.00001064 | mean: -1.00003993 | std: 0.00001376
  droperidol      (True: 1) | min: -1.00004172 | max: -0.99997881 | mean: -0.99999692 | std: 0.00000756
  ibutilide       (True: 0) | min: -1.00026168 | max: -0.99976965 | mean: -1.00016425 | std: 0.00006813
  loratadine      (True: 2) | min: -1.00000530 | max: -0.99999918 | mean: -0.99999996 | std: 0.00000031
  metoprolol      (True: 2) | min: -1.00013946 | max: -0.99997497 | mean: -1.00004970 | std: 0.00002152
  nifedipine      (True: 2) | min: -1.00078315 | max: -1.00009666 | mean: -1.00038108 | std: 0.00007896
  nitrendipine    (True: 2) | min: -1.00019103 | max: -1.00000037 | mean: -1.00004612 | std: 0.00003376
  pimozide        (True: 1) | min: -1.00005605 | max: -0.99999849 | mean: -1.00001719 | std: 0.00000964
  risperidone     (True: 1) | min: -1.00000413 | max: -0.99999036 | mean: -0.99999626 | std: 0.00000223
  tamoxifen       (True: 2) | min: -1.00003245 | max: -0.99999105 | mean: -0.99999768 | std: 0.00000233
  vandetanib      (True: 0) | min: -1.00023142 | max: -0.99999390 | mean: -1.00004060 | std: 0.00002470
  Overlapping pairs: 108
  Up to 10 examples: [('nifedipine', 'ibutilide'), ('nifedipine', 'vandetanib'), ('nifedipine', 'nitrendipine'), ('nifedipine', 'disopyramide'), ('nifedipine', 'metoprolol'), ('nifedipine', 'clozapine'), ('nifedipine', 'domperidone'), ('ibutilide', 'vandetanib'), ('ibutilide', 'nitrendipine'), ('ibutilide', 'disopyramide')]
  Can the drugs be perfectly ordered/separated by vrest alone? NO, ranges overlap.

--- Analysis for qNet ---

TRAINING (12 drugs):
  bepridil        (True: 0) | min: 0.60963152 | max: 0.78312519 | mean: 0.67061432 | std: 0.02248789
  chlorpromazine  (True: 1) | min: 0.90852976 | max: 1.00286979 | mean: 0.95050213 | std: 0.01203509
  cisapride       (True: 1) | min: 0.83097629 | max: 0.91612448 | mean: 0.86903980 | std: 0.01334274
  diltiazem       (True: 2) | min: 0.99828024 | max: 1.11226018 | mean: 1.04121891 | std: 0.01312346
  dofetilide      (True: 0) | min: 0.65681796 | max: 0.77087402 | mean: 0.71809150 | std: 0.01711188
  mexiletine      (True: 2) | min: 1.25617316 | max: 1.46590634 | mean: 1.33712078 | std: 0.02833518
  ondansetron     (True: 1) | min: 0.85153773 | max: 0.94130412 | mean: 0.89255000 | std: 0.01482913
  quinidine       (True: 0) | min: 0.56935407 | max: 0.96164652 | mean: 0.75666481 | std: 0.05235604
  ranolazine      (True: 2) | min: 1.03652267 | max: 1.15947114 | mean: 1.10046635 | std: 0.01837183
  sotalol         (True: 0) | min: 0.74618322 | max: 0.85835150 | mean: 0.79880369 | std: 0.01730718
  terfenadine     (True: 1) | min: 0.81336146 | max: 1.01041030 | mean: 0.88839963 | std: 0.02770740
  verapamil       (True: 2) | min: 0.88001384 | max: 1.11181264 | mean: 1.01616339 | std: 0.03243114
  Overlapping pairs: 29
  Up to 10 examples: [('quinidine', 'bepridil'), ('quinidine', 'dofetilide'), ('quinidine', 'sotalol'), ('quinidine', 'terfenadine'), ('quinidine', 'cisapride'), ('quinidine', 'ondansetron'), ('quinidine', 'verapamil'), ('quinidine', 'chlorpromazine'), ('bepridil', 'dofetilide'), ('bepridil', 'sotalol')]
  Can the drugs be perfectly ordered/separated by qNet alone? NO, ranges overlap.

TESTING (16 drugs):
  astemizole      (True: 1) | min: 0.77726326 | max: 0.95474003 | mean: 0.86605355 | std: 0.02722702
  azimilide       (True: 0) | min: 0.58086912 | max: 0.75807863 | mean: 0.66095362 | std: 0.02958177
  clarithromycin  (True: 1) | min: 0.87506664 | max: 0.97938615 | mean: 0.92497944 | std: 0.01640865
  clozapine       (True: 1) | min: 0.89443533 | max: 1.05148493 | mean: 0.95614152 | std: 0.02251308
  disopyramide    (True: 0) | min: 0.73285375 | max: 0.97321495 | mean: 0.86177949 | std: 0.03241795
  domperidone     (True: 1) | min: 0.56816128 | max: 0.84087148 | mean: 0.67064130 | std: 0.03809737
  droperidol      (True: 1) | min: 0.78365423 | max: 0.95903221 | mean: 0.87554075 | std: 0.02864575
  ibutilide       (True: 0) | min: -0.28024665 | max: 0.55038931 | mean: 0.27438249 | std: 0.15280347
  loratadine      (True: 2) | min: 0.99595311 | max: 1.01130062 | mean: 0.99958419 | std: 0.00066337
  metoprolol      (True: 2) | min: 0.88256390 | max: 1.15413561 | mean: 1.03645183 | std: 0.03745703
  nifedipine      (True: 2) | min: 1.05955642 | max: 1.35185130 | mean: 1.19086677 | std: 0.03599517
  nitrendipine    (True: 2) | min: 0.99992261 | max: 1.10536558 | mean: 1.03651804 | std: 0.01693221
  pimozide        (True: 1) | min: 0.51755948 | max: 0.83820907 | mean: 0.66568216 | std: 0.07014941
  risperidone     (True: 1) | min: 0.90041314 | max: 0.99879618 | mean: 0.96214761 | std: 0.02041734
  tamoxifen       (True: 2) | min: 0.93556295 | max: 1.00230366 | mean: 0.98182018 | std: 0.00997647
  vandetanib      (True: 0) | min: 0.41148763 | max: 0.75911122 | mean: 0.52195372 | std: 0.03813209
  Overlapping pairs: 54
  Up to 10 examples: [('ibutilide', 'vandetanib'), ('ibutilide', 'pimozide'), ('vandetanib', 'pimozide'), ('vandetanib', 'domperidone'), ('vandetanib', 'azimilide'), ('vandetanib', 'disopyramide'), ('pimozide', 'domperidone'), ('pimozide', 'azimilide'), ('pimozide', 'disopyramide'), ('pimozide', 'astemizole')]
  Can the drugs be perfectly ordered/separated by qNet alone? NO, ranges overlap.

--- Analysis for carest ---

TRAINING (12 drugs):
  bepridil        (True: 0) | min: 0.98300000 | max: 1.02000000 | mean: 1.00902950 | std: 0.00331800
  chlorpromazine  (True: 1) | min: 0.98900000 | max: 1.01000000 | mean: 1.00209350 | std: 0.00424379
  cisapride       (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00998000 | std: 0.00044688
  diltiazem       (True: 2) | min: 0.98300000 | max: 1.00000000 | mean: 0.99892800 | std: 0.00181341
  dofetilide      (True: 0) | min: 1.01000000 | max: 1.01000000 | mean: 1.01000000 | std: 0.00000000
  mexiletine      (True: 2) | min: 0.94500000 | max: 1.01000000 | mean: 0.99096000 | std: 0.00861223
  ondansetron     (True: 1) | min: 0.99200000 | max: 1.01000000 | mean: 1.00518000 | std: 0.00527973
  quinidine       (True: 0) | min: 0.96300000 | max: 1.02000000 | mean: 1.00647450 | std: 0.00799658
  ranolazine      (True: 2) | min: 0.98200000 | max: 1.02000000 | mean: 1.01029500 | std: 0.00346886
  sotalol         (True: 0) | min: 1.00000000 | max: 1.02000000 | mean: 1.01016000 | std: 0.00140548
  terfenadine     (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00948000 | std: 0.00222083
  verapamil       (True: 2) | min: 0.93900000 | max: 0.99800000 | mean: 0.97242750 | std: 0.00806674
  Overlapping pairs: 61
  Up to 10 examples: [('verapamil', 'mexiletine'), ('verapamil', 'quinidine'), ('verapamil', 'ranolazine'), ('verapamil', 'bepridil'), ('verapamil', 'diltiazem'), ('verapamil', 'chlorpromazine'), ('verapamil', 'ondansetron'), ('mexiletine', 'quinidine'), ('mexiletine', 'ranolazine'), ('mexiletine', 'bepridil')]
  Can the drugs be perfectly ordered/separated by carest alone? NO, ranges overlap.

TESTING (16 drugs):
  astemizole      (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00922500 | std: 0.00267450
  azimilide       (True: 0) | min: 1.01000000 | max: 1.02000000 | mean: 1.01987000 | std: 0.00113302
  clarithromycin  (True: 1) | min: 0.98800000 | max: 1.01000000 | mean: 1.00634250 | std: 0.00512614
  clozapine       (True: 1) | min: 0.97500000 | max: 1.01000000 | mean: 1.00091550 | std: 0.00383454
  disopyramide    (True: 0) | min: 0.97500000 | max: 1.02000000 | mean: 1.01081800 | std: 0.00676149
  domperidone     (True: 1) | min: 1.00000000 | max: 1.02000000 | mean: 1.01005500 | std: 0.00102347
  droperidol      (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00944000 | std: 0.00229979
  ibutilide       (True: 0) | min: 0.95900000 | max: 1.02000000 | mean: 0.99036300 | std: 0.00838346
  loratadine      (True: 2) | min: 1.00000000 | max: 1.00000000 | mean: 1.00000000 | std: 0.00000000
  metoprolol      (True: 2) | min: 0.98500000 | max: 1.01000000 | mean: 1.00297950 | std: 0.00530756
  nifedipine      (True: 2) | min: 0.83600000 | max: 0.98200000 | mean: 0.91675400 | std: 0.01732303
  nitrendipine    (True: 2) | min: 0.95800000 | max: 1.00000000 | mean: 0.99118550 | std: 0.00776210
  pimozide        (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00997000 | std: 0.00054704
  risperidone     (True: 1) | min: 1.00000000 | max: 1.01000000 | mean: 1.00003500 | std: 0.00059072
  tamoxifen       (True: 2) | min: 0.99300000 | max: 1.00000000 | mean: 0.99998400 | std: 0.00026604
  vandetanib      (True: 0) | min: 0.96200000 | max: 1.02000000 | mean: 1.00412700 | std: 0.00617835
  Overlapping pairs: 107
  Up to 10 examples: [('nifedipine', 'nitrendipine'), ('nifedipine', 'ibutilide'), ('nifedipine', 'vandetanib'), ('nifedipine', 'clozapine'), ('nifedipine', 'disopyramide'), ('nitrendipine', 'ibutilide'), ('nitrendipine', 'vandetanib'), ('nitrendipine', 'clozapine'), ('nitrendipine', 'disopyramide'), ('nitrendipine', 'metoprolol')]
  Can the drugs be perfectly ordered/separated by carest alone? NO, ranges overlap.

==========================================
> RETRACTED. The three bullets below are superseded. Direct recomputation from source data
> (see the overlap tables in this file, and docs/STEP10_VREST_OVERLAP_RECHECK.md) established
> 40 overlapping vrest pairs of 66 on the training split and 108 of 120 on the test split.
> vrest is the LEAST separable feature examined, not the most. Permuting it changes on average
> 1.0 of 16 drug-level predictions, not zero. Authoritative statement: docs/STORYLINE.md Sec. 7.
- vrest perfectly separates the drugs into non-overlapping ranges at extremely precise decimal places.
- The Random Forest base learner splits on vrest massively, isolating extremely fine-grained thresholds (up to 8 decimal places).
- Despite its high SHAP rank, permuting vrest changes zero predictions at the drug level, meaning its splits are structurally redundant copies of information rather than unique deciders.

--- COMPARISON WITH STEP8 TEXT ---
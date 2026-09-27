# Noise audit on 69,504 true (S1, target) pairs from 20,000 sampled S1 records

## Frequency of each pattern (share of true pairs)

| pattern | all | S2 | S3 |
|---|---:|---:|---:|
| name_exact | 4.70% | 4.87% | 4.54% |
| name_exact_casefold | 14.60% | 14.80% | 14.42% |
| name_same_tokens_ordered | 26.01% | 25.47% | 26.52% |
| name_same_token_set | 32.72% | 32.37% | 33.04% |
| name_nonlatin | 7.16% | 9.27% | 5.18% |
| name_domain | 5.12% | 5.12% | 5.12% |
| name_all_upper | 11.42% | 20.55% | 2.87% |
| name_all_lower | 7.07% | 6.84% | 7.28% |
| name_accent_injected | 6.84% | 6.43% | 7.22% |
| name_leet_digit | 2.32% | 2.28% | 2.37% |
| name_leading_junk | 2.61% | 2.74% | 2.48% |
| name_brackets | 6.35% | 6.29% | 6.41% |
| name_hash_number | 0.35% | 0.35% | 0.36% |
| name_pipe_suffix | 0.33% | 0.30% | 0.36% |
| name_dup_token | 4.33% | 4.53% | 4.14% |
| name_double_space | 11.20% | 11.41% | 11.01% |
| name_subset | 22.25% | 24.54% | 20.10% |
| name_superset | 11.28% | 9.57% | 12.87% |
| name_token_overlap0 | 2.92% | 2.45% | 3.35% |
| addr_empty | 4.58% | 4.67% | 4.49% |
| addr_null_literal | 2.45% | 2.49% | 2.40% |
| addr_nonlatin | 9.26% | 9.56% | 8.98% |
| addr_exact | 2.15% | 0.00% | 4.16% |
| addr_same_token_set | 12.79% | 19.20% | 6.79% |
| addr_reordered | 4.59% | 6.72% | 2.60% |
| addr_first_num_equal | 66.58% | 66.37% | 66.78% |
| addr_first_num_lead0 | 3.36% | 3.43% | 3.29% |
| addr_num_missing | 6.45% | 6.70% | 6.23% |
| addr_all_upper | 30.16% | 62.35% | 0.01% |
| country_mismatch | 0.00% | 0.00% | 0.00% |

## Scripts in target names
{'LATIN': 64530, 'DEVANAGARI': 2798, 'TELUGU': 452, 'KANNADA': 390, 'TAMIL': 351, 'BENGALI': 330, 'GUJARATI': 286, 'MALAYALAM': 205, 'ORIYA': 85, 'GURMUKHI': 77}

## Scripts in target addresses
{'LATIN': 63067, 'DEVANAGARI': 3686, 'KANNADA': 548, 'TAMIL': 518, 'TELUGU': 450, 'BENGALI': 419, 'GUJARATI': 379, 'MALAYALAM': 225, 'GURMUKHI': 116, 'ORIYA': 96}

## Tokens added to names (target has, S1 lacks), top 60
[('ltd', 2273), ('center', 2156), ('services', 1485), ('inc', 954), ('l', 773), ('service', 753), ('co', 673), ('c', 624), ('llc', 621), ('the', 616), ('corporation', 584), ('partners', 570), ('corp', 551), ('a', 486), ('limited', 435), ('lp', 390), ('incorporated', 379), ('mr', 371), ('smt', 355), ('shri', 339), ('s', 338), ('dr', 334), ('m', 332), ('sri', 330), ('as', 287), ('formerly', 261), ('dba', 254), ('id', 226), ('k', 203), ('p', 203), ('d', 195), ('b', 193), ('company', 182), ('llp', 145), ('trading', 134), ('f', 132), ('and', 119), ('doing', 105), ('business', 104), ('aka', 99), ('t', 86), ('nee', 77), ('enterprises', 76), ('fka', 76), ('known', 76), ('labs', 72), ('sys', 63), ('one', 57), ('lnc', 51), ('c0m', 36), ('lndia', 34), ('commission', 33), ('5ervices', 27), ('district', 25), ('pc', 22), ('council', 21), ('foundation', 17), ('federation', 16), ('authority', 16), ('trust', 15)]

## Tokens dropped from names, top 60
[('limited', 4907), ('llc', 3493), ('inc', 2318), ('private', 1944), ('ltd', 1308), ('c', 548), ('llp', 412), ('india', 362), ('group', 344), ('corp', 321), ('associates', 306), ('pc', 301), ('p', 293), ('pvt', 272), ('care', 265), ('l', 263), ('co', 235), ('partners', 231), ('clinic', 223), ('center', 217), ('and', 213), ('pllc', 212), ('lp', 201), ('corporation', 184), ('solutions', 166), ('company', 163), ('services', 160), ('d', 144), ('health', 134), ('technologies', 117), ('brothers', 113), ('industries', 112), ('trading', 111), ('medicine', 109), ('of', 102), ('s', 99), ('global', 99), ('enterprises', 98), ('foundation', 97), ('sons', 95), ('trust', 92), ('technology', 85), ('ventures', 84), ('specialists', 83), ('consultants', 82), ('institute', 76), ('holdings', 76), ('consultancy', 73), ('systems', 72), ('school', 70), ('society', 69), ('foods', 67), ('international', 67), ('traders', 64), ('digital', 61), ('physicians', 58), ('construction', 56), ('products', 56), ('business', 55), ('home', 55)]

## Leading junk
[('#', 347), ('@', 297), ('--', 244), ('...', 230), ('>>', 230), ('***', 226), ('(', 114), ('[', 81), ('&', 21), ('+', 3), ('... #', 3), ('<', 2), ('*** #', 2), ('[(', 2), ('([', 1), ('>> @', 1), ('# [', 1), ('[[[', 1), ('-- #', 1), ('[[', 1)]

## Trailing patterns (w=word, 9=digit)
[('[w]', 1790), ('(w)', 1741), ('| w.w.w', 226), ('(w: 99999)', 203), ('#99999', 174), ('[w.]', 83), ('(w.)', 62), ('[[w]]', 37), ('(w) (w)', 32), ('([w])', 32), ('[(w)]', 31), ('((w))', 27), ('[w.w.w.]', 25), ('(w: 9999)', 25), ('(w.w.w.)', 23), ('[w-w]', 22), ('[w.w.]', 22), ('(w,)', 21), ('(w) w (w)', 20), ('[w,]', 19)]

## Tokens added to addresses, top 60
[('rd', 3886), ('st', 3794), ('dr', 3493), ('ave', 2808), ('mh', 2116), ('no', 1954), ('texas', 1797), ('null', 1700), ('ln', 1642), ('new', 1607), ('north', 1471), ('york', 1442), ('carolina', 1336), ('dl', 1310), ('ohio', 1236), ('door', 1228), ('h', 1224), ('city', 1130), ('virginia', 1069), ('a', 1055), ('illinois', 1043), ('1', 1031), ('ct', 946), ('2', 923), ('arizona', 888), ('indiana', 819), ('tennessee', 809), ('ka', 803), ('up', 754), ('massachusetts', 745), ('tn', 723), ('maryland', 654), ('box', 645), ('po', 645), ('wb', 642), ('n', 623), ('washington', 594), ('gj', 579), ('pmb', 573), ('california', 573), ('d', 544), ('alabama', 520), ('c', 516), ('hn', 494), ('minnesota', 491), ('cdp', 487), ('tg', 483), ('block', 477), ('wisconsin', 477), ('saint', 459), ('oregon', 459), ('township', 436), ('kentucky', 426), ('hr', 424), ('plot', 416), ('cir', 413), ('pl', 403), ('arkansas', 372), ('missouri', 364), ('utah', 333)]

## Examples: names with zero token overlap (not native script, not domain)
- `Empire Molecular Inc.` -> `#empiremolecular` | `8909 Scott Hill Drive, Catlettsburg, KY` -> `8909 SCOTT HILL DRIVE, CATLETTSBURG, KY`
- `Royal Foods Private Limited` -> `Jaxdrex` | `Ground Floor, Vile Parle Mahaveer Chsl., 99 Lajpatrai Road, Vile Parle (West), Mumbai, Maharashtra, G1, Mumbai City` -> `G1, Ground Floor, Vile Parle Mahaveer Chsl., 99 Lajpatrai Road, Vile Parle (West), Mumbai, Mumbai City, Maharashtra`
- `Cheslie Moguel Creative Robinhood P.C.` -> `#CHESLIEMOGUEL` | `TN, 119 Fritts Lane, Harriman` -> `HARRIMAN, 119 FRITTS LANE, TN`
- `Behavioral Health Group LP` -> `Gilddrex` | `102 E 4th Ave, Lisbon, IA` -> `102-B East 4th Ave, Lisbon, Iowa`
- `Shivganga Solar Private Limited` -> `#shivgangasolar` | `0N-8/209 B Sundar Purnewada Vns, Varanasi, Uttar Pradesh` -> `0N-8/209 B SUNDAR PURNEWADA VNS, VARANASI, Uttar Pradesh`
- `Alpha Care Pvt Ltd` -> `Smt Quofaye` | `H No.4, Street No. 1, Durga Puri Extension, Delhi` -> `DELHI, STREET NO. 1, Delhi, DURGA PURI EXTENSION, H NO.4`
- `1 800 Movers` -> `#1800` | `11 Parkway Circle, Eastchester, NY` -> `Parkway Circle, Eastchester, New York`
- `Great Services Private Limited` -> `HALOXYLOARC` | `H.N. Ss-4/395, Vibhav Khand, Lucknow, Uttar Pradesh` -> `H.N. SS-4/395, LUCKNOW, उत्तर प्रदेश`
- `Taylor Clean Neighborhood` -> `Yumasyn` | `7032 290 Highway Service Road, Austin, TX` -> `7032 290 HIHGWAY SERVICE ROAD, AUSTIN, TX`
- `Smart Blue Tech Private Limited` -> `#smartblue` | `Plot No.150, Road No.10 Jubulee Hills, Hyderabad, Telangana` -> `H.no ##150, Hyderabad, Road No.10 Jubulee Hills, TG`
- `Mccurry's Investments` -> `Zetagildsyn` | `542 Lehr Avenue, Ada, OH` -> `542 LEHR AVE, ADA, OH`
- `Suresh Broadcasters Private Limited` -> `SBP` | `C/Osanjay Singh, H.No-176, Ward-7 Porsa Gandhi Nagar, Porsa, Morena, Madhya Pradesh` -> `H.NO 78 C/OSANJAY SINGH, PORSA, MORENA, Madhya Pradesh`
- `Fabiano Montessori School Inc` -> `N0vizeta` | `324 Sawyer Road, Greene, ME` -> `324 Saywer Rd, Greene, Maine`
- `Ghaziabad Patra Pvt. Ltd.` -> `Ectohalojax` | `Lgf 32A Devika Chamber Rdc Rajnagar, Ghaziabad, Uttar Pradesh` -> `Morta, उत्तर प्रदेश, Lgf 32A Devika Chamber Rdc Rajnagar, Ghaziabad`
- `First Projects` -> `M/s YUMARIZA` | `No. 55/222, Second Floor, Thekkanath Building, Near South Over Bridge, Ernakulam, Kerala` -> `NO. 55/222, SECOND FLOOR, THEKKANATH BUILDING, NEAR SOUTH OVER BRIDGE, ERNAKULAM, Kerala`
- `Jrk Projects (India) Center` -> `Belofayeiri` | `H No. 302, Vill Harpur, Machhagar Post- Mujdiha, Hata, Kushinagar, Uttar Pradesh` -> `H NO. 302, VILL HARPUR, MACHHAGAR POST- MUJDIHA, HATA, KUSHINAGAR, उत्तर प्रदेश`
- `Jrk Projects (India) Center` -> `Keloorbi` | `H No. 302, Vill Harpur, Machhagar Post- Mujdiha, Hata, Kushinagar, Uttar Pradesh` -> `H.no 0302, Vill Harpur, Machhagar Post- Mujdiha, Hata, Bankata,teh.hata, UP`
- `Office of Environmental Protection` -> `Drexkor` | `8411 Gabrielino Court, Rancho Cucamonga, CA` -> `08411 Gabrielino Court, Rancho Cucamonga Townshp, California`
- `Pincus and Horton LLC` -> `Vioaria` | `1518 Hackberry Heights Drive, Richmond, TX` -> `001518 HACKBERRY HEIGHTS DR, TX, RICHMOND`
- `Projects Kalanidhi Edge Group` -> `UMBRASYNLYRA` | `111, Maruthi Complexraj Bhavan Road Somajiguda, Hyderabad, Telangana` -> `#111, HYDERABAD, తెలంగాణ`
- `Universal Robotics Services LLC` -> `Lumumbranyla` | `14090 Millmac Road, TX, Conroe` -> `14090 MILLMAC RD, CONROE, TX`
- `Bastion Inc.` -> `#BASTI0N` | `3123 Eden Road, Ellston, IA` -> `EDEN ROAD, null, UNION, IA`
- `Ace Kogyo LLC` -> `@acekogyo` | `9014 Us 22, Clarksville, OH` -> `9014 Us 22, Vernon Twp, Ohio`
- `AXZ Devcon Private Limited` -> `DREXBELONOVI` | `Plot No. 15, 16 & 17, Khasra No. 341, House No. 2841, Mouja Wanadongri, Hingna, Nagpur, Maharashtra` -> `NO 15, NAGPUR, Maharashtra`
- `Imperial Kuries Pvt Ltd` -> `Calofaye` | `House No 443, Main Mathura Road Bhogal, Jungpura, New Delhi, South Delhi, Delhi` -> `House No 443, Main Mathura Road Bhogal, Jungpura, New Delhi, South Delhi, Delhi`

## Examples: domain names
- `Life Investments` -> `lifeinvestments.com`
- `India Constructions Private Limited` -> `ipconstructions.com`
- `Vip Public School` -> `Shri vippub1icschool.com`
- `Royal Foods Private Limited` -> `royalfoods.com`
- `Raj & Co Clinic` -> `Smt clinicraj.com`
- `Mireles and Chisholm Bank LLC` -> `mireleschisholmbank.com`
- `NKR Credit Pvt Ltd` -> `nkrcredit.com`
- `AV Farm Private Limited` -> `avfarm.com - 2554699740`
- `AV Farm Private Limited` -> `avfarm.com`
- `Family Associates` -> `familyassociates.com`
- `Family Associates` -> `familyassociates.com`
- `Studio 3 Tattoo` -> `studio3tattoo.com`
- `Studio 3 Tattoo` -> `studio3tattoo.com`
- `Green Retail Solutions Inc` -> `grsolutions.com`
- `Seven Gaushala Private Limited` -> `privatesevengaushala.com`

## Examples: native-script names
- `Lotus Marketing Private Limited` -> `ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | addr `G.S.RESIDENCY, SITE NO.8, FLAT NO.004, 2ND CROSS KEMBATHALLI MAIN ROAD, GOTTIGERE, BANNER, GHATTA RD, BANGALORE, Karnataka`
- `Lotus Marketing Private Limited` -> `ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | addr `G.S.RESIDENCY, SITE NO.8, FLAT NO.004, 2ND CROSS KEMBATHALLI MAIN ROAD, GOTTIGERE, BANNER, GHATTA RD, BANGALORE, ಕರ್ನಾಟಕ`
- `Lotus Marketing Private Limited` -> `Lotus Marketing ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | addr `G.S.RESIDENCY, SITE NO.8, FLAT NO.004, 2ND CROSS KEMBATHALLI MAIN ROAD, GOTTIGERE, BANNER, GHATTA RD, BANGALORE, Karnataka`
- `Lotus Marketing Private Limited` -> `ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | addr `H.no 3-568 G.s.residency, Site No.8, Flat No.004, 2Nd Cross Kembathalli Main Road, Gottigere, Banner, Ghatta Rd, Bengaluru, Bangalore, ಕರ್ನಾಟಕ`
- `Life Investments` -> `लाइफ इन्वेस्टमेंट्स` | addr `Maharashtra, OFFICE NO 212, THANE`
- `Life Investments` -> `Life इन्वेस्टमेंट्स` | addr `OFFICE NO 212, PLOT NO 20, THE GREAT EASTERNGALLERIA, THANE, महाराष्ट्र`
- `Jay Finance` -> `जय फाइनेंस` | addr `C/O RAJEEV SHARMA X-8, JABALPUR, मध्य प्रदेश`
- `Maa Services LLP` -> `मां सर्विसेज एलएलपी` | addr `78C, NEW DELHI, WEST DELHI, Delhi`
- `Shiva Construction` -> `शिवा कंस्ट्रक्शन` | addr `MUMBAI (SUBURBAN), MUMBAI, Maharashtra, C-206 JAIAM ARCADE`
- `Shiva Construction` -> `शिवा Construction` | addr `C-2-08 Jainam Arcade, Mumbai, MH`
- `Royal Foods Private Limited` -> `रॉयल फूड्स प्राइवेट लिमिटेड` | addr `G1, Ground Floor, Vile Parle Mahaveer Chsl., 99 Lajpatrai Road, Vile Parle (West), Mumbai City, Mumbai, महाराष्ट्र`
- `Royal Foods Private Limited` -> `रॉयल फूड्स प्राइवेट लिमिटेड` | addr `G1, Mumbai, Mumbai City, MH`
- `Fortune Foundation Private Limited` -> `फॉर्च्यून फाउंडेशन प्राइवेट लिमिटेड` | addr `DELHI, NORTH WEST DELHI, UNIT NO. 204`
- `Fortune Foundation Private Limited` -> `फॉर्च्यून फाउंडेशन प्राइवेट लिमिटेड` | addr `North Delhi, DL, Unit No. 204`
- `Fortune Foundation Private Limited` -> `फॉर्च्यून फाउंडेशन प्राइवेट लिमिटेड` | addr `Unit No. 204, Delhi, North Delhi, DL`
"""DEV ONLY - builds a small fake dataset in the challenge's file format so the
pipeline can be tested end-to-end before the real data is available.
Never used to train or tune the submitted model.

  python dev/make_synthetic.py --out dev/fake_resource
"""
import argparse
import random
from pathlib import Path

R = random.Random(7)

US = dict(
    first=["Blue", "Summit", "Golden", "Pacific", "Liberty", "Eagle", "Maple", "Harbor", "Pioneer",
           "Evergreen", "Silver", "Northstar", "Granite", "Redwood", "Keystone", "Lakeside", "Apex"],
    second=["Dental", "Logistics", "Bakery", "Auto Repair", "Consulting", "Hardware", "Pharmacy",
            "Realty", "Plumbing", "Coffee", "Printing", "Fitness", "Insurance", "Electric", "Foods"],
    legal=["Inc", "LLC", "Corp", "Co", "Ltd", ""],
    streets=["Main", "Oak", "Maple", "Washington", "Lake", "Hill", "Park", "Pine", "Cedar", "Elm"],
    stype=[("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"), ("Drive", "Dr")],
    cities=[("Austin", "TX", "Texas"), ("Denver", "CO", "Colorado"), ("Seattle", "WA", "Washington"),
            ("Chicago", "IL", "Illinois"), ("Boston", "MA", "Massachusetts"), ("Phoenix", "AZ", "Arizona")],
)
IN = dict(
    first=["Shree", "Sai", "Ganesh", "Lakshmi", "Balaji", "Krishna", "Durga", "Mahalaxmi", "Om",
           "Jai Hind", "Bharat", "Ambika", "Shiv", "Annapurna", "Navkar", "Jyoti"],
    second=["Traders", "Enterprises", "Medical Store", "Textiles", "Sweets", "Electronics",
            "Hardware", "Jewellers", "Motors", "Agencies", "Industries", "Kirana Store"],
    legal=["Pvt Ltd", "Private Limited", "LLP", "Ltd", ""],
    areas=["MG Road", "Station Road", "Gandhi Nagar", "Laxmi Chowk", "Sector 14", "Shivaji Marg",
           "Nehru Colony", "Main Bazaar", "Rajendra Nagar"],
    landmarks=["Near SBI ATM", "Opp Bus Stand", "Behind Hanuman Mandir", "Near Railway Station",
               "Opposite Post Office"],
    cities=[("Pune", "Maharashtra", "411"), ("Jaipur", "Rajasthan", "302"), ("Indore", "Madhya Pradesh", "452"),
            ("Lucknow", "Uttar Pradesh", "226"), ("Nagpur", "Maharashtra", "440"), ("Surat", "Gujarat", "395")],
)
FR = dict(
    first=["Boulangerie", "Pharmacie", "Garage", "Cabinet", "Librairie", "Brasserie", "Atelier", "Fromagerie"],
    second=["Dupont", "Martin", "Lefebvre", "du Centre", "de la Gare", "Saint Michel", "Moreau", "des Arts"],
    legal=["SARL", "SAS", "SA", "EURL", ""],
    streets=["de la Paix", "Victor Hugo", "Jean Jaures", "de la Republique", "Pasteur", "Gambetta"],
    stype=[("Rue", "R"), ("Avenue", "Av"), ("Boulevard", "Bd"), ("Place", "Pl")],
    cities=[("Paris", "75"), ("Lyon", "69"), ("Marseille", "13"), ("Toulouse", "31"), ("Nantes", "44")],
)
TRANSLIT = {"Shree": "Shri", "Chowk": "Chauk", "Laxmi": "Lakshmi", "Nagar": "Nagr", "Bazaar": "Bazar",
            "Jewellers": "Jewelers", "Mahalaxmi": "Mahalakshmi", "Sai": "Saai"}
ABBR = {"Street": "St", "Avenue": "Ave", "Road": "Rd", "Boulevard": "Blvd", "Drive": "Dr",
        "Corporation": "Corp", "Private Limited": "Pvt Ltd", "Limited": "Ltd", "Company": "Co",
        "Enterprises": "Ent", "and": "&", "Rue": "R", "Boulevard ": "Bd ", "Saint": "St"}


def entity(country, i):
    if country == "US":
        d = US
        name = f"{R.choice(d['first'])} {R.choice(d['second'])}"
        legal = R.choice(d["legal"])
        city, st, stfull = R.choice(d["cities"])
        full_t, short_t = R.choice(d["stype"])
        addr = dict(no=str(R.randint(10, 9999)), street=f"{R.choice(d['streets'])} {full_t}",
                    city=city, state=st, state_full=stfull, pc=f"{R.randint(10000, 99999)}", lm="")
    elif country == "India":
        d = IN
        name = f"{R.choice(d['first'])} {R.choice(d['second'])}"
        legal = R.choice(d["legal"])
        city, state, pre = R.choice(d["cities"])
        addr = dict(no=f"{R.randint(1, 300)}", street=R.choice(d["areas"]), city=city, state=state,
                    state_full=state, pc=f"{pre}{R.randint(0, 999):03d}", lm=R.choice(d["landmarks"]))
    else:
        d = FR
        name = f"{R.choice(d['first'])} {R.choice(d['second'])}"
        legal = R.choice(d["legal"])
        city, dep = R.choice(d["cities"])
        full_t, _ = R.choice(d["stype"])
        addr = dict(no=str(R.randint(1, 150)), street=f"{full_t} {R.choice(d['streets'])}", city=city,
                    state="", state_full="", pc=f"{dep}{R.randint(0, 999):03d}", lm="")
    return dict(name=name, legal=legal, addr=addr, country=country)


def fmt_addr(a, country, noisy):
    parts_no, street = a["no"], a["street"]
    lm = a["lm"] if (a["lm"] and (not noisy or R.random() < 0.6)) else ""
    state = a["state"] if not noisy or R.random() < 0.7 else ""
    if noisy and R.random() < 0.3 and a["state_full"]:
        state = a["state_full"]
    pc = a["pc"] if not noisy or R.random() < 0.75 else ""
    if country == "France":
        s = f"{parts_no} {street}, {pc} {a['city']}"
    elif country == "India":
        s = ", ".join(x for x in [f"{parts_no}, {street}" if R.random() < 0.8 or not noisy else street,
                                  lm, a["city"], state, pc] if x)
    else:
        s = ", ".join(x for x in [f"{parts_no} {street}", a["city"], f"{state} {pc}".strip()] if x)
    if noisy and R.random() < 0.3 and country == "India" and pc and len(pc) == 6:
        s = s.replace(pc, pc[:3] + " " + pc[3:])
    return s


def typo(s):
    if len(s) < 4:
        return s
    i = R.randrange(1, len(s) - 1)
    op = R.random()
    if op < 0.33:
        return s[:i] + s[i + 1:]
    if op < 0.66:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + R.choice("aeiourstn") + s[i:]


def noisy_copy(e):
    name = e["name"]
    legal = e["legal"]
    if R.random() < 0.4:
        legal = R.choice(["", legal, {"Inc": "Incorporated", "Corp": "Corporation", "Co": "Company",
                                      "Ltd": "Limited", "Pvt Ltd": "Private Limited", "LLC": "L.L.C.",
                                      "SARL": "S.A.R.L."}.get(legal, legal)])
    for a, b in list(TRANSLIT.items()) + list(ABBR.items()):
        if a in name and R.random() < 0.4:
            name = name.replace(a, b)
    if R.random() < 0.25:
        name = typo(name)
    if R.random() < 0.1:
        w = name.split()
        if len(w) > 1:
            w[0], w[1] = w[1], w[0]
            name = " ".join(w)
    if R.random() < 0.15:
        name = name.upper()
    full = f"{name} {legal}".strip()
    if R.random() < 0.05:
        full = f"{full} DBA {R.choice(['Express', 'Plus', 'Hub'])}"
    addr = fmt_addr(e["addr"], e["country"], noisy=True)
    for a, b in list(ABBR.items()) + list(TRANSLIT.items()):
        if a in addr and R.random() < 0.5:
            addr = addr.replace(a, b)
    if R.random() < 0.2:
        addr = typo(addr)
    return full, addr


def build(split, n_s1, countries, out):
    ents = [entity(R.choice(countries), i) for i in range(int(n_s1 * 1.3))]
    # chain branches: same name, different place -> distinct entities (hard negatives)
    for _ in range(n_s1 // 8):
        base = R.choice(ents)
        e = entity(base["country"], 0)
        e["name"], e["legal"] = base["name"], base["legal"]
        ents.append(e)
    R.shuffle(ents)
    s1_ents, extra = ents[:n_s1], ents[n_s1:]
    rows = {1: [], 2: [], 3: []}
    gt = []
    cnt = {1: 0, 2: 0, 3: 0}

    def add(src, name, addr, country):
        cnt[src] += 1
        eid = f"S{src}-{cnt[src]:05d}"
        rows[src].append((eid, name, addr, country))
        return eid

    for e in s1_ents:
        sid = add(1, f"{e['name']} {e['legal']}".strip(), fmt_addr(e["addr"], e["country"], False), e["country"])
        matches = []
        k = R.choices([0, 1, 2, 3, 4], weights=[35, 30, 20, 10, 5])[0]
        for _ in range(k):
            src = R.choice([2, 3])
            n, a = noisy_copy(e)
            matches.append(add(src, n, a, e["country"]))
        gt.append((sid, ",".join(matches)))
    for e in extra:  # records in S2/S3 with no S1 counterpart
        for _ in range(R.choice([1, 1, 2])):
            n, a = noisy_copy(e)
            add(R.choice([2, 3]), n, a, e["country"])
    d = Path(out) / "dataset" / split
    d.mkdir(parents=True, exist_ok=True)
    for src in (1, 2, 3):
        recs = rows[src][:]
        if src > 1:
            R.shuffle(recs)
        with open(d / f"{split}_source{src}.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for r in recs:
                f.write("\t".join(r) + "\n")
    gt_path = d / "train_ground_truth.tsv" if split == "train" else Path(out) / "dev_test_ground_truth.tsv"
    if True:
        with open(gt_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for sid, m in gt:
                f.write(f"{sid}\t{m}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dev/fake_resource")
    a = ap.parse_args()
    build("train", 3000, ["US", "India"], a.out)
    build("test", 1500, ["US", "India", "France"], a.out)
    print("fake dataset written to", a.out)


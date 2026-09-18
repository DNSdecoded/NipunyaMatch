"""Generate 12 synthetic resume PDFs (one two-column, one image-only) plus golden labels."""
import json
from pathlib import Path

import pymupdf as fitz

OUT = Path("tests/fixtures/resumes")
OUT.mkdir(parents=True, exist_ok=True)

PEOPLE = [
    ("Asha Rao", "asha.rao@example.com", "+91 98765 43210",
     ["Python", "FastAPI", "Docker", "PostgreSQL", "TensorFlow"], "2018-06", None, "B.Tech Computer Science", 85),
    ("Ben Carter", "ben.carter@example.com", "+1 415 555 0101",
     ["Python", "Django", "PostgreSQL"], "2016-01", "2023-12", "BSc Software Engineering", 70),
    ("Chen Wei", "chen.wei@example.com", "+86 138 0000 0000",
     ["Java", "Spring", "Kubernetes", "AWS"], "2015-03", None, "M.S. Computer Science", 55),
    ("Dana Ortiz", "dana.ortiz@example.com", "+34 600 000 000",
     ["Python", "scikit-learn", "Pandas", "SQL"], "2021-09", None, "MSc Data Science", 65),
    ("Eli Novak", "eli.novak@example.com", "+420 600 000 000",
     ["Python", "FastAPI", "Docker", "AWS", "PostgreSQL"], "2019-02", None, "B.Eng", 88),
    ("Fatima Khan", "fatima.khan@example.com", "+92 300 0000000",
     ["JavaScript", "React", "Node.js"], "2020-05", None, "BS Computer Science", 30),
    ("Grace Obi", "grace.obi@example.com", "+234 800 000 0000",
     ["Python", "Flask", "Docker", "MySQL"], "2017-08", None, "B.Tech IT", 62),
    ("Hiro Tanaka", "hiro.tanaka@example.com", "+81 90 0000 0000",
     ["Go", "Docker", "Kubernetes", "PostgreSQL"], "2014-04", None, "Bachelor of Engineering", 50),
    ("Ivy Larsen", "ivy.larsen@example.com", "+45 20 00 00 00",
     ["Python", "PyTorch", "FastAPI", "Docker"], "2022-01", None, "PhD Machine Learning", 80),
    ("Jonas Meyer", "jonas.meyer@example.com", "+49 151 00000000",
     ["C++", "Python", "Linux"], "2012-10", None, "Diploma Informatik", 40),
    ("Kavya Iyer", "kavya.iyer@example.com", "+91 90000 00000",
     ["Python", "FastAPI", "PostgreSQL", "ML", "Docker", "Kubernetes"], "2019-07", None, "B.Tech CSE", 92),
    ("Leo Santos", "leo.santos@example.com", "+55 11 90000 0000",
     ["PHP", "Laravel", "MySQL"], "2018-03", None, "Associate Degree", 20),
]


def body(name: str, email: str, phone: str, skills: list[str], start: str,
         end: str | None, degree: str) -> list[str]:
    return [
        name, f"{email} | {phone}", "",
        "SKILLS", ", ".join(skills), "",
        "EXPERIENCE", f"Software Engineer, Example Corp ({start} - {end or 'Present'})",
        f"Built and maintained services using {', '.join(skills[:3])}.", "",
        "EDUCATION", f"{degree}, Example University, 2016",
    ]


def write_pdf(path: Path, lines: list[str], two_column: bool = False, image_only: bool = False) -> None:
    doc = fitz.open()
    page = doc.new_page()
    if two_column:
        half = len(lines) // 2
        for i, ln in enumerate(lines[:half]):
            page.insert_text((40, 60 + 16 * i), ln, fontsize=10)
        for i, ln in enumerate(lines[half:]):
            page.insert_text((320, 60 + 16 * i), ln, fontsize=10)
    else:
        for i, ln in enumerate(lines):
            page.insert_text((60, 60 + 16 * i), ln, fontsize=10)
    if image_only:
        pix = page.get_pixmap(dpi=150)
        img_doc = fitz.open()
        p = img_doc.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, pixmap=pix)
        img_doc.save(path)
        return
    doc.save(path)


labels: dict[str, dict[str, object]] = {}
for i, (name, email, phone, skills, start, end, degree, fit) in enumerate(PEOPLE):
    fname = f"{i + 1:02d}_{name.split()[0].lower()}.pdf"
    write_pdf(OUT / fname, body(name, email, phone, skills, start, end, degree),
              two_column=(i == 2), image_only=(i == 9))
    labels[fname] = {"name": name, "email": email, "phone": phone, "skills": skills,
                     "start_date": start, "end_date": end, "degree": degree, "llm_fit_score": fit}

Path("tests/fixtures/golden").mkdir(parents=True, exist_ok=True)
Path("tests/fixtures/golden/labels.json").write_text(json.dumps(labels, indent=2))
Path("tests/fixtures/golden/ranking.json").write_text(json.dumps([
    "11_kavya.pdf", "05_eli.pdf", "01_asha.pdf", "09_ivy.pdf", "02_ben.pdf", "04_dana.pdf",
    "07_grace.pdf", "03_chen.pdf", "08_hiro.pdf", "10_jonas.pdf", "06_fatima.pdf", "12_leo.pdf",
], indent=2))
print(f"wrote {len(labels)} resumes")

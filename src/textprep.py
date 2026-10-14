import re

# Lines produced by issue-form placeholders and the show_versions() block
NOISE_LINE = re.compile(
    r"^(_no response_|system:|python:|executable:|machine:|python dependencies:|"
    r"sklearn:|pip:|setuptools:|numpy:|scipy:|cython:|pandas:|matplotlib:|joblib:|"
    r"threadpoolctl:|built with openmp|user_api:|internal_api:|num_threads:|"
    r"prefix:|filepath:|version:|threading_layer:|architecture:)",
    re.I,
)


def strip_template(issue):
    keep = []
    for line in issue["body"].splitlines():
        s = line.strip()
        if s.startswith(("#", ">", "```")):      # headings, banner, code fences
            continue
        if NOISE_LINE.match(s) or "site-packages" in s:
            continue
        keep.append(s)
    text = issue["title"] + " " + " ".join(keep)
    return re.sub(r"\s+", " ", text).strip()


def build_text(title, body):
    """Same cleaning as prepare.py: title + body, whitespace collapsed."""
    return re.sub(r"\s+", " ", title + "\n" + (body or "")).strip()
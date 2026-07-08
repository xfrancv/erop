import json
import os
from getpass import getpass
from pathlib import Path

import openreview


FORUM_ID = os.environ.get("OPENREVIEW_FORUM") or input("Forum id from URL: ").strip()
USERNAME = os.environ.get("OPENREVIEW_USERNAME") or input("OpenReview username/email: ").strip()
PASSWORD = os.environ.get("OPENREVIEW_PASSWORD") or getpass("OpenReview password: ")

client = openreview.api.OpenReviewClient(
    baseurl="https://api2.openreview.net",
    username=USERNAME,
    password=PASSWORD,
)

notes = client.get_all_notes(forum=FORUM_ID)

out = []
for note in notes:
    invitations = getattr(note, "invitations", []) or []
    invitation_text = " ".join(invitations).lower()

    # Keep likely review/decision/meta-review notes.
    # You can remove this filter if you want every comment/rebuttal too.
    wanted = any(
        key in invitation_text
        for key in [
            "official_review",
            "review",
            "decision",
            "meta_review",
            "metareview",
        ]
    )

    if not wanted:
        continue

    content = {}
    for key, value in (note.content or {}).items():
        # API v2 usually stores content as {"field": {"value": ...}}
        if isinstance(value, dict) and "value" in value:
            content[key] = value["value"]
        else:
            content[key] = value

    out.append(
        {
            "id": note.id,
            "number": getattr(note, "number", None),
            "cdate": getattr(note, "cdate", None),
            "tcdate": getattr(note, "tcdate", None),
            "invitations": invitations,
            "signatures": getattr(note, "signatures", []),
            "readers": getattr(note, "readers", []),
            "content": content,
        }
    )

Path("openreview_reviews_decision.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

with open("openreview_reviews_decision.md", "w", encoding="utf-8") as f:
    for i, item in enumerate(out, 1):
        title = " / ".join(item["invitations"])
        f.write(f"# Note {i}: {title}\n\n")
        for key, value in item["content"].items():
            f.write(f"## {key}\n\n{value}\n\n")

print(f"Saved {len(out)} notes:")
print("  openreview_reviews_decision.json")
print("  openreview_reviews_decision.md")

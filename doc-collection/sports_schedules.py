"""Save the Steelers, Pirates and Penguins schedules as one line per game.

The scraped schedule pages don't work for retrieval: steelers.com puts every field on its own line
(WEEK 8 / VS / Browns / CBS / ... with no year), and the Pirates and Penguins schedule pages are
rendered with JavaScript. Instead:
    Steelers: the game cards in https://www.steelers.com/schedule/ (data-gametime has the full date)
    Pirates:  MLB's JSON API  https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=134&season=2026
    Penguins: NHL's JSON API  https://api-web.nhle.com/v1/club-schedule-season/PIT/20262027
Output: <host>/schedule-<season>.txt, e.g. www.steelers.com/schedule-2026.txt
"""
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (CMU ANLP course project crawler)"}
EASTERN = ZoneInfo("America/New_York")


def day(d):
    return f"{d:%A, %B} {d.day}, {d.year}"


def clock(d):
    return d.strftime("%I:%M %p").lstrip("0") + " ET"


def text(el):
    return re.sub(r"\s+", " ", el.get_text(" ")).strip() if el else ""


def save(path, url, heading, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"URL: {url}\n\n{heading}\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved   {path} ({len(lines)} games)")


def steelers(season, out_dir):
    url = "https://www.steelers.com/schedule/"
    soup = BeautifulSoup(requests.get(url, headers=HEADERS, timeout=30).content, "html.parser")
    lines, section = [], ""
    # section headings (REGULAR SEASON, PRESEASON) and game cards, in page order
    for el in soup.find_all(lambda t: t.name == "span" and text(t) in ("REGULAR SEASON", "PRESEASON")
                            or "nfl-o-matchup-cards" in (t.get("class") or [])):
        if el.name == "span":
            section = text(el).title()
            continue
        week = text(el.find("strong")) or text(el.select_one(".nfl-o-matchup-cards__date-info"))
        week = re.match(r"WEEK \d+", week).group().title() if re.match(r"WEEK \d+", week) else week
        if "BYE" in text(el):
            lines.append(f"Steelers {section} {week}: bye week")
            continue
        prefix = text(el.select_one(".nfl-o-matchup-cards__team-prefix, .nfl-o-matchup-cards__team-game-location"))
        opponent = text(el.select_one(".nfl-o-matchup-cards__team-full-name"))
        venue = text(el.select_one(".nfl-o-matchup-cards__venue--location"))
        game = f"Pittsburgh Steelers {'vs.' if prefix == 'VS' else 'at'} {opponent}"
        when = datetime.strptime(el["data-gametime"], "%m/%d/%Y %H:%M:%S %z")
        if when.year < 1900:  # 01/01/0001: date not announced yet
            when = "date and time to be determined"
        else:
            when = f"{day(when)}, {clock(when.astimezone(EASTERN))}"
        line = f"Steelers {section} {week}: {game}, {when}, {venue}"
        result = text(el.select_one(".nfl-o-matchup-cards__score--result"))
        if result:
            points = text(el.select_one(".nfl-o-matchup-cards__score--points"))
            status = text(el.select_one(".nfl-o-matchup-cards__date-info--final")).strip("· ")
            outcome = {"W": "won", "L": "lost", "T": "tied"}.get(result, result)
            line += f". Final: Steelers {outcome} {points}" + (" in overtime" if "OT" in status else "")
        else:
            tv = text(el.select_one(".nfl-o-matchup-cards__media-tv--networks"))
            radio = text(el.select_one(".nfl-o-matchup-cards__media-radio--networks"))
            if tv and tv != "TBD":
                line += f". TV: {tv}"
            if radio:
                line += f". Radio: {radio}"
        lines.append(line + ".")
    save(Path(out_dir) / "www.steelers.com" / f"schedule-{season}.txt", url,
         f"Pittsburgh Steelers {season} Schedule and Results", lines)


def pirates(season, out_dir):
    api = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=134&season={season}"
           "&gameType=S,R,F,D,L,W")
    data = requests.get(api, headers=HEADERS, timeout=30).json()
    kinds = {"S": "Spring training", "R": "Regular season", "F": "Wild Card", "D": "Division Series",
             "L": "League Championship Series", "W": "World Series"}
    lines = []
    for date in data["dates"]:
        for g in date["games"]:
            home, away = g["teams"]["home"], g["teams"]["away"]
            pirates_home = home["team"]["id"] == 134
            opponent = (away if pirates_home else home)["team"]["name"]
            start = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")).astimezone(EASTERN)
            when = day(datetime.fromisoformat(g["officialDate"]))
            if not g["status"].get("startTimeTBD"):
                when += f", {clock(start)}"
            game = f"Pittsburgh Pirates {'vs.' if pirates_home else 'at'} {opponent}"
            if g.get("doubleHeader") in ("Y", "S"):
                game += f" (doubleheader game {g['gameNumber']})"
            line = f"{kinds.get(g['gameType'], g['gameType'])}: {game}, {when}, {g['venue']['name']}"
            state = g["status"]["detailedState"]
            if state == "Final" and "score" in home:
                us, them = (home, away) if pirates_home else (away, home)
                outcome = "won" if us.get("isWinner") else "lost"
                line += f". Final: Pirates {outcome} {us['score']}-{them['score']}"
                if g["gameType"] == "R":
                    rec = us["leagueRecord"]
                    line += f" (record {rec['wins']}-{rec['losses']})"
            elif state not in ("Scheduled", "Pre-Game", "Final"):
                line += f". Status: {state}"
            lines.append(line + ".")
    save(Path(out_dir) / "www.mlb.com" / f"pirates__schedule-{season}.txt",
         f"https://www.mlb.com/pirates/schedule/{season}",
         f"Pittsburgh Pirates {season} Schedule and Results", lines)


def penguins(season, out_dir):
    """season: first year, e.g. 2026 for the 2026-27 season."""
    code = f"{season}{season + 1}"
    data = requests.get(f"https://api-web.nhle.com/v1/club-schedule-season/PIT/{code}",
                        headers=HEADERS, timeout=30).json()
    kinds = {1: "Preseason", 2: "Regular season", 3: "Playoffs"}
    name = lambda t: f"{t['placeName']['default']} {t['commonName']['default']}"
    lines = []
    for g in data["games"]:
        home, away = g["homeTeam"], g["awayTeam"]
        pens_home = home["abbrev"] == "PIT"
        opponent = name(away if pens_home else home)
        start = datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00")).astimezone(EASTERN)
        game = f"Pittsburgh Penguins {'vs.' if pens_home else 'at'} {opponent}"
        line = f"{kinds.get(g['gameType'], 'Game')}: {game}, {day(start)}, {clock(start)}, {g['venue']['default']}"
        if g["gameState"] in ("FINAL", "OFF") and "score" in home:
            us, them = (home, away) if pens_home else (away, home)
            outcome = "won" if us["score"] > them["score"] else "lost"
            line += f". Final: Penguins {outcome} {us['score']}-{them['score']}"
            period = g.get("gameOutcome", {}).get("lastPeriodType")
            if period in ("OT", "SO"):
                line += " in overtime" if period == "OT" else " in a shootout"
        else:
            us_tv = [b["network"] for b in g.get("tvBroadcasts", []) if b["countryCode"] == "US"]
            if us_tv:
                line += f". TV: {', '.join(us_tv)}"
        if g.get("gameScheduleState", "OK") != "OK":
            line += f". Status: {g['gameScheduleState'].lower()}"
        lines.append(line + ".")
    save(Path(out_dir) / "www.nhl.com" / f"penguins__schedule-{season}-{season + 1}.txt",
         "https://www.nhl.com/penguins/schedule",
         f"Pittsburgh Penguins {season}-{str(season + 1)[2:]} Schedule and Results", lines)


if __name__ == "__main__":
    out_dir = "data/pages"
    steelers(2026, out_dir)
    pirates(2026, out_dir)
    penguins(2026, out_dir)

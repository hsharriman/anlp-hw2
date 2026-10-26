"""Team schedules, rosters and coaching staffs, one line per game or person.

    python sports.py [source ...]     sources: schedules rosters college (default: all)

    schedules  Steelers, Pirates, Penguins. The scraped schedule pages don't work for retrieval:
               steelers.com puts every field on its own line (WEEK 8 / VS / Browns / ... with no
               year) and the Pirates and Penguins pages are rendered with JavaScript. Instead:
                 Steelers: game cards in https://www.steelers.com/schedule/ (data-gametime has the date)
                 Pirates:  MLB's JSON API https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=134
                 Penguins: NHL's JSON API https://api-web.nhle.com/v1/club-schedule-season/PIT/20262027
    rosters    Pirates 40-man roster and coaches (MLB API, with birthplace and MLB debut), Penguins
               roster (NHL API; nhl.com's staff pages need JavaScript, so no coaches), Steelers
               roster (tables on steelers.com; the coaches page is plain text, saved by scrape.py)
    college    Pitt, Robert Morris, Duquesne and Chatham athletics, every sport (SIDEARM Sports sites)
Output: <host>/schedule-<season>.txt, roster-<season>.txt, coaches-<season>.txt,
schedule-<sport>-<season>.txt
"""
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pycountry
import requests
from bs4 import BeautifulSoup

from common import MONTHS, PAGES, clock, day, get, save_doc, soup_of, text

SEASON = 2026  # first year of the season (2026 for the NHL's 2026-27)
EASTERN = ZoneInfo("America/New_York")


def clock_et(d):
    return clock(d) + " ET"


def long_date(iso):
    """2016-08-13 -> August 13, 2016 (rosters give both forms, since questions use either)"""
    d = date.fromisoformat(iso)
    return f"{d:%B} {d.day}, {d.year}"


# --- pro schedules ---------------------------------------------------------------------------------

def steelers_schedule(out_dir, season=SEASON):
    url = "https://www.steelers.com/schedule/"
    lines, section = [], ""
    # section headings (REGULAR SEASON, PRESEASON) and game cards, in page order
    for el in soup_of(url).find_all(lambda t: t.name == "span" and text(t) in ("REGULAR SEASON", "PRESEASON")
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
        when = ("date and time to be determined" if when.year < 1900  # 01/01/0001: not announced yet
                else f"{day(when)}, {clock_et(when.astimezone(EASTERN))}")
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
    save_doc(Path(out_dir) / "www.steelers.com" / f"schedule-{season}.txt", url,
             f"Pittsburgh Steelers {season} Schedule and Results", lines, unit="games")


def pirates_schedule(out_dir, season=SEASON):
    api = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId=134&season={season}&gameType=S,R,F,D,L,W"
    kinds = {"S": "Spring training", "R": "Regular season", "F": "Wild Card", "D": "Division Series",
             "L": "League Championship Series", "W": "World Series"}
    lines = []
    for game_day in get(api).json()["dates"]:
        for g in game_day["games"]:
            home, away = g["teams"]["home"], g["teams"]["away"]
            pirates_home = home["team"]["id"] == 134
            opponent = (away if pirates_home else home)["team"]["name"]
            start = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")).astimezone(EASTERN)
            when = day(datetime.fromisoformat(g["officialDate"]))
            if not g["status"].get("startTimeTBD"):
                when += f", {clock_et(start)}"
            game = f"Pittsburgh Pirates {'vs.' if pirates_home else 'at'} {opponent}"
            if g.get("doubleHeader") in ("Y", "S"):
                game += f" (doubleheader game {g['gameNumber']})"
            line = f"{kinds.get(g['gameType'], g['gameType'])}: {game}, {when}, {g['venue']['name']}"
            state = g["status"]["detailedState"]
            if state == "Final" and "score" in home:
                us, them = (home, away) if pirates_home else (away, home)
                line += f". Final: Pirates {'won' if us.get('isWinner') else 'lost'} {us['score']}-{them['score']}"
                if g["gameType"] == "R":
                    line += f" (record {us['leagueRecord']['wins']}-{us['leagueRecord']['losses']})"
            elif state not in ("Scheduled", "Pre-Game", "Final"):
                line += f". Status: {state}"
            lines.append(line + ".")
    save_doc(Path(out_dir) / "www.mlb.com" / f"pirates__schedule-{season}.txt",
             f"https://www.mlb.com/pirates/schedule/{season}", f"Pittsburgh Pirates {season} Schedule and Results",
             lines, unit="games")


def penguins_schedule(out_dir, season=SEASON):
    data = get(f"https://api-web.nhle.com/v1/club-schedule-season/PIT/{season}{season + 1}").json()
    kinds = {1: "Preseason", 2: "Regular season", 3: "Playoffs"}
    name = lambda t: f"{t['placeName']['default']} {t['commonName']['default']}"
    lines = []
    for g in data["games"]:
        home, away = g["homeTeam"], g["awayTeam"]
        pens_home = home["abbrev"] == "PIT"
        start = datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00")).astimezone(EASTERN)
        game = f"Pittsburgh Penguins {'vs.' if pens_home else 'at'} {name(away if pens_home else home)}"
        line = f"{kinds.get(g['gameType'], 'Game')}: {game}, {day(start)}, {clock_et(start)}, {g['venue']['default']}"
        if g["gameState"] in ("FINAL", "OFF") and "score" in home:
            us, them = (home, away) if pens_home else (away, home)
            line += f". Final: Penguins {'won' if us['score'] > them['score'] else 'lost'} {us['score']}-{them['score']}"
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
    save_doc(Path(out_dir) / "www.nhl.com" / f"penguins__schedule-{season}-{season + 1}.txt",
             "https://www.nhl.com/penguins/schedule",
             f"Pittsburgh Penguins {season}-{str(season + 1)[2:]} Schedule and Results", lines, unit="games")


def schedules(out_dir=PAGES):
    steelers_schedule(out_dir)
    pirates_schedule(out_dir)
    penguins_schedule(out_dir)


# --- pro rosters -----------------------------------------------------------------------------------

def pirates_roster(out_dir, season=SEASON):
    api = "https://statsapi.mlb.com/api/v1/teams/134"
    lines = []
    for r in sorted(get(f"{api}/roster?rosterType=40Man&season={season}&hydrate=person").json()["roster"],
                    key=lambda r: r["person"]["lastName"]):
        p = r["person"]
        line = f"{p['fullName']}, Pittsburgh Pirates {r['position']['name'].lower()}"
        if r.get("jerseyNumber"):
            line += f", number {r['jerseyNumber']}"
        line += f" ({r['status']['description'].lower()})."
        born = ", ".join(x for x in (p.get("birthCity"), p.get("birthStateProvince"), p.get("birthCountry")) if x)
        if p.get("birthDate"):
            line += f" Born {long_date(p['birthDate'])} ({p['birthDate']})" + (f" in {born}." if born else ".")
        if p.get("mlbDebutDate"):
            line += f" MLB debut: {long_date(p['mlbDebutDate'])} ({p['mlbDebutDate']})."
        line += f" Bats {p['batSide']['description'].lower()}, throws {p['pitchHand']['description'].lower()}."
        line += f" Height {p['height']}, weight {p['weight']} lb."
        lines.append(line)
    save_doc(Path(out_dir) / "www.mlb.com" / f"pirates__roster-{season}.txt", "https://www.mlb.com/pirates/roster/40-man",
             f"Pittsburgh Pirates {season} 40-Man Roster", lines, unit="people")
    lines = [f"{c['person']['fullName']} is the {c['job']} for the Pittsburgh Pirates"
             + (f" (number {c['jerseyNumber']})." if c.get("jerseyNumber") else ".")
             for c in get(f"{api}/coaches?season={season}").json()["roster"]]
    save_doc(Path(out_dir) / "www.mlb.com" / f"pirates__coaches-{season}.txt", "https://www.mlb.com/pirates/team/coaches",
             f"Pittsburgh Pirates {season} Coaching Staff", lines, unit="people")


def country(code):
    c = pycountry.countries.get(alpha_3=code)
    return getattr(c, "common_name", None) or (c.name if c else code)


def penguins_roster(out_dir, season=SEASON):
    data = get(f"https://api-web.nhle.com/v1/roster/PIT/{season}{season + 1}").json()
    positions = {"C": "center", "L": "left wing", "R": "right wing", "D": "defenseman", "G": "goaltender"}
    lines = []
    for group in ("forwards", "defensemen", "goalies"):
        for p in sorted(data[group], key=lambda p: p["lastName"]["default"]):
            line = (f"{p['firstName']['default']} {p['lastName']['default']}, Pittsburgh Penguins "
                    f"{positions.get(p['positionCode'], p['positionCode'])}")
            if p.get("sweaterNumber"):
                line += f", number {p['sweaterNumber']}"
            born = [p.get("birthCity", {}).get("default"), p.get("birthStateProvince", {}).get("default"),
                    country(p["birthCountry"]) if p.get("birthCountry") else None]
            line += f". Born {long_date(p['birthDate'])} in {', '.join(x for x in born if x)}."
            shoots = {"L": "left", "R": "right"}.get(p.get("shootsCatches"), "")
            if shoots:
                line += f" {'Catches' if group == 'goalies' else 'Shoots'} {shoots}."
            feet, inches = divmod(p["heightInInches"], 12)
            line += f" Height {feet}' {inches}\", weight {p['weightInPounds']} lb."
            lines.append(line)
    save_doc(Path(out_dir) / "www.nhl.com" / f"penguins__roster-{season}-{season + 1}.txt",
             "https://www.nhl.com/penguins/roster", f"Pittsburgh Penguins {season}-{str(season + 1)[2:]} Roster",
             lines, unit="people")


def steelers_roster(out_dir, season=SEASON):
    url = "https://www.steelers.com/team/players-roster/"
    lines = []
    for table in soup_of(url).find_all("table"):
        group = text(table.find_previous(["h2", "h3", "h4"]))  # Active, Practice Squad, ...
        columns = [text(th) for th in table.find("thead").find_all("th")]
        for tr in table.find("tbody").find_all("tr"):
            row = dict(zip(columns, (text(td) for td in tr.find_all("td"))))
            exp = row.get("Exp", "")
            exp = "rookie" if exp == "R" else f"{exp} years of NFL experience" if exp else ""
            line = (f"{row['Player']}, Pittsburgh Steelers {row.get('Pos', '')}, number {row.get('#', '')} "
                    f"({group}). Height {row.get('HT', '')}, weight {row.get('WT', '')} lb, age {row.get('Age', '')}")
            line += f", {exp}" if exp else ""
            line += f", college: {row['College']}." if row.get("College") else "."
            lines.append(line)
    save_doc(Path(out_dir) / "www.steelers.com" / f"roster-{season}.txt", url,
             f"Pittsburgh Steelers {season} Roster", lines, unit="people")


def rosters(out_dir=PAGES):
    pirates_roster(out_dir)
    penguins_roster(out_dir)
    steelers_roster(out_dir)


# --- college athletics -----------------------------------------------------------------------------
# SIDEARM Sports sites render a schedule in one of two layouts:
#   classic  <li class="sidearm-schedule-game sidearm-schedule-home-game ..."> per game; the opponent
#            link's aria-label has the full date and time ("Bethany on October 9 5 PM")
#   nextgen  game cards that show only some games, so the site's text version is used instead:
#            /sports/<sport>/schedule/text, a table with Date, Time, At (Home/Away/Neutral),
#            Opponent, Location, Tournament and Result columns
# /sports/<sport>/schedule shows the current season. Sports come from the site's menu plus a list of
# common sport slugs (the nextgen menu doesn't link every sport); missing ones are skipped.

SCHOOLS = [("pittsburghpanthers.com", "Pitt"), ("rmucolonials.com", "Robert Morris"),
           ("goduquesne.com", "Duquesne"), ("gochathamcougars.com", "Chatham")]
FIRST_GAME_DAY = date(2026, 8, 1)  # the assignment asks about events from October 2026
COMMON_SLUGS = [f"{g}{s}" for s in ["basketball", "soccer", "cross-country", "swimming-and-diving",
                                    "track-and-field", "lacrosse", "golf", "tennis", "ice-hockey"]
                for g in ("mens-", "womens-")] + ["football", "baseball", "softball", "womens-volleyball",
                                                  "wrestling", "gymnastics", "womens-rowing"]


def sport_name(slug):
    """womens-ice-hockey -> women's ice hockey"""
    return re.sub(r"^(wo)?mens-", lambda m: m.group(0)[:-2] + "'s ", slug).replace("-", " ")


def season_years(title):
    """'2026-27 Men's Hockey Schedule' -> (2026, 2027); '2026 Women's Soccer Schedule' -> (2026, 2026)"""
    m = re.search(r"(20\d\d)(?:[-–](\d\d))?", title)
    if not m:
        return None
    return int(m.group(1)), 2000 + int(m.group(2)) if m.group(2) else int(m.group(1))


def game_date(month_day, years):
    """'Oct 9' with the season's years -> date; fall months fall in the first year"""
    m = re.match(r"([A-Za-z]{3})[a-z]*\.? (\d{1,2})", month_day.strip())
    if not m:
        return None
    month = MONTHS.index(m.group(1).title()) + 1
    return date(years[0] if month >= 7 else years[1], month, int(m.group(2)))


def classic_games(soup, years):
    for li in soup.select("li.sidearm-schedule-game"):
        classes = li.get("class", [])
        side = "home" if "sidearm-schedule-home-game" in classes else (
            "away" if "sidearm-schedule-away-game" in classes else "neutral")
        row = li.select_one(".sidearm-schedule-game-row-desktop") or li
        label = (row.select_one(".sidearm-schedule-game-opponent-name a") or {}).get("aria-label", "")
        at = re.search(r"(\d{1,2}(?::\d\d)? ?[AP]\.?M\.?|TBA|TBD)\s*$", label, re.I)
        yield dict(date=game_date(text(row.select_one(".sidearm-schedule-game-opponent-date")), years), side=side,
                   opponent=text(row.select_one(".sidearm-schedule-game-opponent-name")),
                   time=at.group(1) if at else "", location=text(row.select_one(".sidearm-schedule-game-location")),
                   result=text(row.select_one(".sidearm-schedule-game-result")).rstrip(","))


def text_games(soup, years):
    table = next((t for t in soup.find_all("table")
                  if [text(th) for th in t.find_all("th")][:4] == ["Date", "Time", "At", "Opponent"]), None)
    for row in table.find_all("tr")[1:] if table else []:
        cells = [text(td) for td in row.find_all("td")]
        if len(cells) < 5:
            continue
        when, at, side, opponent, location = cells[:5]
        tournament = cells[5] if len(cells) > 5 and cells[5] != "-" else ""
        yield dict(date=game_date(when, years), side=side.lower(), opponent=opponent, time=at,
                   location=location + (f" ({tournament})" if tournament else ""),
                   result=cells[6] if len(cells) > 6 and cells[6] != "-" else "")


def describe_game(school, sport, g):
    vs = {"home": "home game vs", "away": "away game at"}.get(g["side"], "neutral-site game vs")
    line = f"{school} {sport}: {vs} {g['opponent']} on {day(g['date'])}"
    if g["time"] and not g["time"].lower().startswith("final"):
        line += f" at {g['time']}"
    if g["location"]:
        line += f", {g['location']}"
    line += "."
    if g["result"] and re.match(r"[WLT]\b", g["result"]):
        line += f" Result: {g['result']}."
    return line


def college_school(host, school, out_dir):
    base = f"https://{host}"
    menu = soup_of(f"{base}/sports/{'womens-soccer' if host == 'gochathamcougars.com' else 'football'}/schedule")
    slugs = sorted({m.group(1) for a in menu.find_all("a", href=True)
                    if (m := re.fullmatch(r"(?:https://" + re.escape(host) + r")?/sports/([a-z-]+)/schedule/?",
                                          a["href"]))} | set(COMMON_SLUGS))
    for slug in slugs:
        time.sleep(1)
        url = f"{base}/sports/{slug}/schedule"
        try:
            response = get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            if response.status_code == 200 and not soup.select("li.sidearm-schedule-game"):
                response = get(f"{url}/text")
                soup = BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"failed  {url}: {e}")
            continue
        title = next((text(h) for h in soup.find_all(["h1", "h2", "title"]) if "Schedule" in text(h)), "")
        years = season_years(title)
        if response.status_code != 200 or not years:
            print(f"skip    {url} ({response.status_code}, title {title!r})")
            continue
        games = list(classic_games(soup, years)) or list(text_games(soup, years))
        games = [g for g in games if g["date"] and g["date"] >= FIRST_GAME_DAY and g["opponent"]]
        if not games:
            print(f"skip    {url} (no games from {FIRST_GAME_DAY})")
            continue
        season = f"{years[0]}" if years[0] == years[1] else f"{years[0]}-{str(years[1])[2:]}"
        save_doc(Path(out_dir) / host / f"schedule-{slug}-{season}.txt", response.url,
                 f"{school} {sport_name(slug)} {season} schedule",
                 [describe_game(school, sport_name(slug), g) for g in games], unit="games")


def college(out_dir=PAGES):
    """Output: <host>/schedule-<sport>-<season>.txt, e.g. pittsburghpanthers.com/schedule-womens-basketball-2026-27.txt"""
    for host, school in SCHOOLS:
        college_school(host, school, out_dir)


SOURCES = {"schedules": schedules, "rosters": rosters, "college": college}

if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        if name not in SOURCES:
            sys.exit(__doc__)
        SOURCES[name]()

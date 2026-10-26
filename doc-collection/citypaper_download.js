// Download all Pittsburgh City Paper event listing pages as one HTML file for `python events.py citypaper`.
//
// community.pghcitypaper.com blocks scripts, so this runs in your own browser session:
//   1. Open https://community.pghcitypaper.com/events?to=2026-12-31&from=2026-10-23
//   2. Open DevTools (Cmd+Option+J), paste this whole file into the Console, press Enter.
//   3. Move the downloaded citypaper_events.html into doc-collection/raw_pages/citypaper/
// Pages are fetched one at a time with a pause in between, like clicking through them.
// If no download appears, the HTML is kept in window.citypaperHTML: run copy(window.citypaperHTML)
// in the console, then in a terminal: pbpaste > doc-collection/raw_pages/citypaper/citypaper_events.html
(async () => {
  const FROM = "2026-10-23";
  const TO = "2026-12-31";
  const DELAY_MS = 3000;

  const pageUrl = (n) => `${location.origin}/events?to=${TO}&from=${FROM}&page=${n}`;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const getPage = async (n) => {
    const res = await fetch(pageUrl(n), { credentials: "include" });
    const doc = new DOMParser().parseFromString(await res.text(), "text/html");
    const main = doc.querySelector("main");
    const events = main ? main.querySelectorAll("h3 a[href*='/events/']").length : 0;
    return { status: res.status, doc, main, events };
  };

  const parts = [];
  try {
    const first = await getPage(1);
    if (!first.main || !first.events) {
      throw new Error(`page 1: HTTP ${first.status}, ${first.events} events; is this tab on community.pghcitypaper.com?`);
    }
    // number of pages = ceil(results / events per page); the promoted card is outside <main>
    const m = first.doc.body.textContent.match(/([\d,]+)\s*results/);
    const pages = m ? Math.ceil(parseInt(m[1].replace(/,/g, ""), 10) / first.events) : 100;
    console.log(`${m ? m[1] : "?"} results, ${first.events} per page -> ${pages} pages`);
    parts.push(first.main.outerHTML);

    for (let n = 2; n <= pages; n++) {
      await sleep(DELAY_MS);
      const page = await getPage(n);
      if (!page.main || !page.events) {
        console.warn(`page ${n}: HTTP ${page.status}, ${page.events} events; stopping`);
        break;
      }
      parts.push(page.main.outerHTML);
      console.log(`page ${n}/${pages}: ${page.events} events`);
    }
  } catch (err) {
    console.error(err);
  }
  if (!parts.length) return;

  // one file: the <main> of every page, in order (the parser reads day headings and cards in order)
  const html = `<!doctype html><html><body>\n${parts.join("\n")}\n</body></html>`;
  window.citypaperHTML = html;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  a.download = "citypaper_events.html";
  document.body.appendChild(a); // Firefox only downloads from links that are in the page
  a.click();
  a.remove();
  console.log(`${parts.length} pages (${html.length} characters) -> citypaper_events.html. ` +
              "No download? Run copy(window.citypaperHTML) and paste it into a file.");
})();

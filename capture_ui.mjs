import puppeteer from "puppeteer";

(async () => {
    try {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();

        page.on('console', msg => console.log('PAGE LOG:', msg.text()));
        page.on('pageerror', error => console.error('PAGE ERROR:', error.message));
        page.on('requestfailed', request => {
            if (request.failure()) {
                console.log('REQUEST FAILED:', request.url(), request.failure().errorText);
            }
        });

        await page.goto("http://localhost:5173/");

        // click the button to start the simulation
        // button class is btn-primary, wait for it
        await page.waitForSelector(".btn-primary");
        await page.click(".btn-primary");
        console.log("Clicked logic to start...");

        // wait for a bit to let it run
        await new Promise(r => setTimeout(r, 15000));
        await browser.close();
    } catch (e) {
        console.error(e);
    }
})();

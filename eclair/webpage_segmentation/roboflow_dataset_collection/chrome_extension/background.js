function sanitizeFilename(filename) {
    return filename.replace(/[^a-z0-9]/gi, '_');
}

function takeSnapshot() {
    const uniqueId = Math.random().toString(36).substr(2, 9);      
    // 1) Get webpage state as JSON...
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        const tabId = tabs[0].id;
        const tabTitle = sanitizeFilename(tabs[0].title);
        chrome.tabs.sendMessage(tabId, { action: 'takeSnapshot', data: null }, function(vocXmlAnnotation) {
            // Download annotation XML file
            const encodedUri = encodeURIComponent(vocXmlAnnotation);
            const dataUri = 'data:text/xml;charset=utf-8,' + encodedUri;
            chrome.downloads.download({
                url: dataUri,
                filename: `${tabTitle} - ${uniqueId}.xml`,
            });
            // 2) Take screenshot...    
            chrome.tabs.captureVisibleTab(null, { format: "png" }, function(screenshotUrl) {
                // Download screenshot image as PNG
                chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                    const tabTitle = sanitizeFilename(tabs[0].title);
                    chrome.downloads.download({
                        url: screenshotUrl,
                        filename: `${tabTitle} - ${uniqueId}.png`,
                    });
                });
            })
        })
    });
}

chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.action === "takeSnapshot") {
        takeSnapshot();
    }
});

chrome.commands.onCommand.addListener((command) => {
    if (command === "takeSnapshot") {
        takeSnapshot();
    }
});
  
function triggerAction(action, data = null) {
  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    chrome.tabs.sendMessage(
      tabs[0].id,
      { action: action, data: data },
      function (response) {
        console.log(response);
      }
    );
  });
}

document.getElementById("snapshotBtn").addEventListener("click", function () {
  chrome.runtime.sendMessage({ action: "takeSnapshot" });
});

document.getElementById("toggleBoundingBoxesBtn").addEventListener("click", function () {
  triggerAction("toggleBoundingBoxes");
});

document.getElementById("toggleLabelsBtn").addEventListener("click", function () {
  triggerAction("toggleLabels");
});

document.getElementById("printStateBtn").addEventListener("click", function () {
  triggerAction("printState");
});



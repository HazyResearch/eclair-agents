chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log(`Fire action: ${request.action}`);
  if (request.action === "takeSnapshot") {
    // Make sure we have a clean webpage
    removeBoundingBoxes();
    removeLabels();
    // Generate JSON state
    const elementsData = getElementsData();
    const state = generateJSONState(elementsData);
    // Convert JSON -> VOC XML
    const vocXmlAnnotation = jsonStateToVOCXML(state);
    sendResponse(vocXmlAnnotation);
  } else if (request.action === "toggleBoundingBoxes") {
    toggleBoundingBoxes();
  } else if (request.action === "toggleLabels") {
    toggleLabels();
  } else if (request.action === "printState") {
    printState();
  } else {
    console.log(`Invalid action: ${request.action}`);
  }
  return true;
});

let isLabelsShowing = false;
let isBoundingBoxesShowing = false;

function getElementsData() {
  // Returns a list of elements that are (1) visible (2) in accessibility tree (3) non-overlapping
  const allElements = getAllMatchingElements("body *");
  const accessibilityElements = filterOutNonAccessibleElements(allElements);
  const filteredElements = filterOutOverlappingBoxes( accessibilityElements, 0.5 );
  return filteredElements;
}
function toggleLabels() {
  const elementsData = getElementsData();
  if (isLabelsShowing) {
    removeLabels();
  } else {
    addLabels(elementsData);
  }
  isLabelsShowing = !isLabelsShowing;
}
function toggleBoundingBoxes() {
  const elementsData = getElementsData();
  if (isBoundingBoxesShowing) {
    removeBoundingBoxes();
  } else {
    drawBoundingBoxes(elementsData);
  }
  isBoundingBoxesShowing = !isBoundingBoxesShowing;
}

function printState() {
  const elementsData = getElementsData();
  const json = generateJSONState(elementsData);
  console.log(json);
}
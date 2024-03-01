/**
 * Helper functions for selecting/filtering elements
 */

const INTERESTING_TAGS = [
  "input",
  "textarea",
  "button",
  "select",
  "label",
  "a",
  "img",
];

function isVisible(element) {
  // Returns true if `element` is visible in the viewport, false otherwise.
  const rect = element.getBoundingClientRect();
  const isNonZeroArea = rect.width > 0 && rect.height > 0;
  const isNonTrivialArea = rect.width > 3 && rect.height > 3; // element should be at least 3x3 pixels (filter out 1x1 icons)
  const isNotHidden =
    getComputedStyle(element).display !== "none" &&
    getComputedStyle(element).visibility === "visible";
  const isInViewport =
    rect.bottom > 0 &&
    rect.right > 0 &&
    rect.top < (window.innerHeight || document.documentElement.clientHeight) &&
    rect.left < (window.innerWidth || document.documentElement.clientWidth);
  const isAriaHidden = element.getAttribute("aria-hidden") === "true";
  return (
    isNonZeroArea &&
    isNonTrivialArea &&
    isNotHidden &&
    isInViewport &&
    !isAriaHidden
  );
}

function getXPath(element) {
  // Returns an XPath for an element which describes its hierarchical location in the DOM.
  if (element && element.id) {
    return "//*[@id='" + element.id + "']";
  } else {
    let segments = [];
    while (element && element.nodeType === 1) {
      let siblingIndex = 1;
      for (
        let sibling = element.previousSibling;
        sibling;
        sibling = sibling.previousSibling
      ) {
        if (sibling.nodeType === 1 && sibling.tagName === element.tagName)
          siblingIndex++;
      }
      let tagName = element.tagName.toLowerCase();
      let segment =
        tagName +
        (element.previousElementSibling || element.nextElementSibling
          ? "[" + siblingIndex + "]"
          : "");
      segments.unshift(segment);
      element = element.parentNode;
    }
    return segments.length ? "/" + segments.join("/") : null;
  }
}

function getParents(element) {
  // Get list of parent elements for `element`.
  let parents = [];
  while (element) {
    parents.unshift(element.tagName + (element.id ? "#" + element.id : ""));
    element = element.parentElement;
  }
  return parents;
}

function getElementByXPath(xpath) {
  // Returns the unique element matching the `xpath` expression.
  const result = document.evaluate(
    xpath,
    document,
    null,
    XPathResult.FIRST_ORDERED_NODE_TYPE,
    null
  );
  return result.singleNodeValue;
}

function getAllMatchingElements(querySelector) {
  // Given a `querySelector`, returns a list of objects describing the elements matched by document.querySelectorAll(querySelector).
  const elementsData = [];

  document.querySelectorAll(querySelector).forEach((element) => {
    if (isVisible(element)) {
      const rect = element.getBoundingClientRect();
      const xpath = getXPath(element);

      elementsData.push({
        // Element ID
        element: element,
        xpath: xpath,
        // Position
        x: rect.x,
        y: rect.y,
        height: rect.height,
        width: rect.width,
        // Attributes
        text: element.innerText,
        role: element.getAttribute("role"),
        type: element.getAttribute("type"),
        tag: element.tagName.toLowerCase(),
        label: element.getAttribute("aria-label"),
        // State
        is_focused: document.activeElement === element,
        is_checked: element.checked ? true : false,
        is_disabled: element.disabled ? true : false,
      });
    }
  });

  return elementsData;
}

/**
 * Helper functions for drawing bounding boxes
 */

// Add bounding box style
const style = document.createElement("style");
style.textContent = `
.llm-agent-box {
    border: 1px solid #e00fdf !important;
}
`;
document.head.appendChild(style);

// Draw boxes around each element
function drawBoundingBoxes(elementsData) {
  var BreakException = {};
  elementsData.forEach((data) => {
    try {
      const $element = getElementByXPath(data.xpath);
      if ($element) {
        $element.classList.add("llm-agent-box");
      } else {
      }
    } catch {
      console.log(data);
      throw BreakException;
    }
  });
}

// Remove bounding boxes
function removeBoundingBoxes() {
  document.querySelectorAll(".llm-agent-box").forEach((element) => {
    element.classList.remove("llm-agent-box");
  });
}

// Define a bounding box as { x: 0, y: 0, width: 0, height: 0 }
function intersectionArea(boxA, boxB) {
  const xOverlap = Math.max(
    0,
    Math.min(boxA.x + boxA.width, boxB.x + boxB.width) -
      Math.max(boxA.x, boxB.x)
  );
  const yOverlap = Math.max(
    0,
    Math.min(boxA.y + boxA.height, boxB.y + boxB.height) -
      Math.max(boxA.y, boxB.y)
  );
  return xOverlap * yOverlap;
}

function boxArea(box) {
  return box.width * box.height;
}

function overlapRatio(boxA, boxB) {
  // Calculate the overlap ratio between two bounding boxes (boxA and boxB)
  const intersection = intersectionArea(boxA, boxB);
  const union = boxArea(boxA) + boxArea(boxB) - intersection;
  return intersection / union;
}
/**
 * Helper functions for drawing labels on bounding boxes
 */

function onLabelMouseOverShowBoundingBox($labelElement) {
  // When a label is moused over, show its corresponding bounding box
  const xpath = $labelElement.attributes["elementsData-xpath"];
  const $element = getElementByXPath(xpath);
  if ($element) {
    $element.classList.add("llm-agent-box");
  }
}
function onLabelMouseOutHideBoundingBox($labelElement) {
  // When a label is moused out, hide its corresponding bounding box
  const xpath = $labelElement.attributes["elementsData-xpath"];
  const $element = getElementByXPath(xpath);
  if ($element) {
    $element.classList.remove("llm-agent-box");
  }
}
function addLabels(elementsData) {
  // Given a list of elements with bounding boxes, add a label above each element on the webpage to visualize labels.
  elementsData.forEach((data, idx) => {
    // Get the element using XPath
    const element = getElementByXPath(data.xpath);
    if (element) {
      // Create a new div for the label
      let labelDiv = document.createElement("div");

      // Set the content and style
      labelDiv.innerText = `${data.label} (${data.type}) | ${truncate(
        data.text,
        10
      )}`;
      labelDiv.style.position = "absolute";
      labelDiv.style.left = data.x + "px";
      labelDiv.style.top = data.y - 9 + "px";
      labelDiv.style.height = "9px";
      labelDiv.style.backgroundColor = "rgba(255, 255, 255, 0.7)"; // semi-transparent white background
      labelDiv.style.border = "1px solid black";
      labelDiv.style.padding = "2px";
      labelDiv.style.fontSize = "8px";
      labelDiv.style.zIndex = "10000";
      labelDiv.classList.add("llm-agent-box-label");
      labelDiv.attributes["elementsData-xpath"] = data.xpath;
      labelDiv.attributes["elementsData-idx"] = idx;

      // Add event listener for hovering over the label
      labelDiv.addEventListener("mouseover", () =>
        onLabelMouseOverShowBoundingBox(labelDiv)
      );
      labelDiv.addEventListener("mouseout", () =>
        onLabelMouseOutHideBoundingBox(labelDiv)
      );

      // Append the label div to the body (or to the specific element if you prefer)
      document.body.appendChild(labelDiv);
    }
  });
}
function removeLabels() {
  document
    .querySelectorAll(".llm-agent-box-label")
    .forEach((element) => element.remove());
}

function generateJSONState(elementsData) {
  // Given a list of elements, generate a JSON string representing the state of the page.
  const filteredElementsData = elementsData
    .filter((data) => {
      // Ignore things off screen
      if (data.x < 0 || data.y < 0) {
        return false;
      }
      // Ignore things with no text, label, etc. that aren't interesting tags
      if (!INTERESTING_TAGS.includes(data.tag)) {
        if (
          data.label === null &&
          data.role === null &&
          (data.text === "" || data.text === null)
        ) {
          return false;
        }
      }
      return true;
    })
    .map((data) => {
      return {
        x: Math.round(data.x),
        y: Math.round(data.y),
        height: Math.round(data.height),
        width: Math.round(data.width),
        // Attributes
        text: data.text,
        label: data.label,
        role: data.role,
        type: data.type,
        tag: data.tag,
        xpath: data.xpath,
        // States
        is_focused: data.is_focused,
        is_checked: data.is_checked,
        is_disabled: data.is_disabled,
      };
    });
  return JSON.stringify(filteredElementsData);
}
function truncate(text, n) {
  return text.length > n ? text.substr(0, n - 1) + "…" : text;
}

function filterOutNonAccessibleElements(elementsData) {
  // Filter out elements that are not accessible
  return elementsData.filter((data) => {
    // Valid if has at least one of...
    const hasRoleAttribute = data.element.getAttribute("role") !== null;
    const hasAriaAttribute =
      Array.from(data.element.attributes).filter((e) =>
        e.name.startsWith("aria-")
      ).length > 0;
    const hasDataAttribute =
      data.element.getAttribute("data-view-component") !== null;
    const isInteractableElement = INTERESTING_TAGS.includes(
      data.element.tagName.toLowerCase()
    );
    const isLink =
      data.element.tagName.toLowerCase() === "a" &&
      data.element.getAttribute("href") !== null;
    const isTextBlock =
      ["span", "p"].includes(data.element.tagName.toLowerCase()) &&
      data.element.innerText.length > 0;
    return (
      hasRoleAttribute ||
      hasAriaAttribute ||
      hasDataAttribute ||
      isInteractableElement ||
      isLink ||
      isTextBlock
    );
  });
}

function filterOutOverlappingBoxes(elementsData, threshold) {
  // `elementsData` must be a list of dicts with keys: x,y,height,width
  const toRemove = new Set();

  for (let i = 0; i < elementsData.length; i++) {
    if (toRemove.has(i)) continue;

    for (let j = i + 1; j < elementsData.length; j++) {
      if (toRemove.has(j)) continue;

      if (overlapRatio(elementsData[i], elementsData[j]) > threshold) {
        // Decide which box to remove.
        // Preferences:
        //  1. Keep element with aria-label
        const ariaLabelI = elementsData[i].label;
        const ariaLabelJ = elementsData[j].label;
        if (ariaLabelI !== null && ariaLabelJ === null) {
          toRemove.add(j);
        } else if (ariaLabelI === null && ariaLabelJ !== null) {
          toRemove.add(i);
          break; // No need to continue this inner loop since i is marked for removal
        } else {
          //  2. Keep smaller element
          const areaI = boxArea(elementsData[i]);
          const areaJ = boxArea(elementsData[j]);
          if (areaI >= areaJ) {
            toRemove.add(i);
            break; // No need to continue this inner loop since i is marked for removal
          } else {
            toRemove.add(j);
          }
        }
      }
    }
  }

  return elementsData.filter((_, index) => !toRemove.has(index));
}

////////////////////////
// Everything above should be copied from `chrome_extension/html_to_state.js`
////////////////////////

const allElements = getAllMatchingElements("body *");
const accessibilityElements = filterOutNonAccessibleElements(allElements);
const filteredElements = filterOutOverlappingBoxes(accessibilityElements, 0.5);
const json = generateJSONState(filteredElements);
return json;

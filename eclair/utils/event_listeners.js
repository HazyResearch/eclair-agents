
if (!window.isEventListenerLoaded) {
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

    function getElementAttributes($element) {
        let rect = { x: 0, y: 0, height: 0, width: 0, }
        if ($element.getBoundingClientRect) {
            rect = $element.getBoundingClientRect();
        }
        return {
            xpath : getXPath($element),
            label : $element.getAttribute('aria-label') || $element.getAttribute('aria-labelledby'),
            type : $element.getAttribute('type'),
            placeholder : $element.getAttribute('placeholder'),
            role : $element.getAttribute('role'),
            tag : $element.tagName.toLowerCase(),
            text : $element.innerText,
            value: $element.value,
            x: rect.x,
            y: rect.y,
            height: rect.height,
            width: rect.width,
        }
    }
    
    // Track last clicked element
    document.addEventListener("mousedown", function(event) {
        console.log('mousedown fired', event.target);
        window.lastMouseDown = {
            x: event.clientX,
            y: event.clientY,
            element: getElementAttributes(event.target)
        };
        localStorage.setItem('lastMouseDown', JSON.stringify(window.lastMouseDown));
    });
    document.addEventListener("mouseup", function(event) {
        console.log('mouseup fired', event.target);
        window.lastMouseUp = {
            x: event.clientX,
            y: event.clientY,
            element: getElementAttributes(event.target),
        }
        localStorage.setItem('lastMouseUp', JSON.stringify(window.lastMouseUp));
    });

    // Track last keystroked input
    document.addEventListener("keydown", function(event) {
        console.log('keydown fired', event.target);
        window.lastKeyDown = {
            key: event.key,
            element: getElementAttributes(event.target),
        }
        localStorage.setItem('lastKeyDown', JSON.stringify(window.lastKeyDown));
    });
    document.addEventListener("keyup", function(event) {
        console.log('keyup fired', event.target);
        window.lastKeyUp = {
            key: event.key,
            element: getElementAttributes(event.target),
        }
        localStorage.setItem('lastKeyUp', JSON.stringify(window.lastKeyUp));
    });

    // Track last scrolled element
    document.addEventListener("scroll", function(event) {
        console.log('scroll fired', event.target);
        window.lastScrolled = {
            xpath : getXPath(event.target),
        }
        localStorage.setItem('lastScrolled', JSON.stringify(window.lastScrolled));
    });

    window.isEventListenerLoaded = true;
}
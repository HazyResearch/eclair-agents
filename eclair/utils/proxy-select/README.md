# proxy-select
Replaces native select with pure HTML based select to allow event handling when select is opened.

# Use case

Primary usage is to control the select even when select is expanded/opened. Normally, when select is opened OS takes control and you don't know where mouse is till an option is not selected.

Also using headless tools like puppeteer to take screenshots, it allows you to capture correct screenshot when select is opened.

Alternatively, you can use libraries like chosen which will allow the same thing but they come with their own API and styling while proxy-select can be included in any project and won't affect existing code. It also uses native JS.

# Usage

Include two files

```
<script src="../proxy-select.js"></script>
<link rel="stylesheet" href="../proxy-select.css" >
```

By default, it will change behavior of all the selects on the page including ones which might be added in later.

# Requirements

This is impemented in native JS and doesn't reuire any other library like JQuery.
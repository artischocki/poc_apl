// Override the default Chainlit favicon with the project favicon
const link = document.querySelector("link[rel='icon']") || document.createElement("link");
link.rel = "icon";
link.type = "image/png";
link.href = "/public/favicon.png";
document.head.appendChild(link);

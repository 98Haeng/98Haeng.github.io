const clock = document.querySelector("h2#clock");

function getClock () {
    const date = new Date();
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    const seconds = String(date.getSeconds()).padStart(2, "0");
    clock.innerText = `${hours}:${minutes}:${seconds}`;
}

// load되자마자 실행하고 매초마다 실행하게끔
getClock();
setInterval(getClock, 1000); // 매 초마다 실행 (실시간)

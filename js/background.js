const images = ["note1.jpg","note2.jpg","note3.jpg","note4.jpg"];

const chosenImage = images[Math.floor(Math.random() * images.length)];

// 새로운 항목을 만들 때 createElement 사용
const bgImage = document.createElement("img");
bgImage.src = `img/${chosenImage}`;

// 이미지를 내부에 추가하기
document.body.appendChild(bgImage);
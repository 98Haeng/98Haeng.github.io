const toDoForm = document.getElementById("todo-form");
const toDoInput = document.querySelector("#todo-form input");
const toDoList = document.getElementById("todo-list");

const TODOS_KEY = "todos";

let toDos = [];

function saveToDos(){
    localStorage.setItem("todos", JSON.stringify(toDos));
}

function deleteToDo(event){
    const li = event.target.parentElement; // 어떤 버튼이 클릭되었는지 알아야 함.
    li.remove();
    toDos = toDos.filter((toDo) => toDo.id !== parseInt(li.id));
    // 지우고 나서 저장
    saveToDos();
}

function paintToDo(newTodo){
    const li = document.createElement("li");
    li.id = newTodo.id; // 데이터베이스에 id 부여
    const span = document.createElement("span");
    span.innerText = newTodo.text;
    const button = document.createElement("button");
    button.innerText = "❌";
    button.addEventListener("click", deleteToDo);
    li.appendChild(span);
    li.appendChild(button);
    toDoList.appendChild(li);// 단, 새로고침하면 사라짐
}

function handleToDoSubmit(event) {
    event.preventDefault();
    const newTodo = toDoInput.value;
    toDoInput.value = ""; // input 즉, 적어야 하는 부분을 비움
    const newTodoObj = {
        text:newTodo,
        id:Date.now(), 
    }
    toDos.push(newTodoObj); // object 형태로 저장하기
    paintToDo(newTodoObj)
    saveToDos();
}

function sayHello(item){
    console.log("this is the turn of ", item);
}

toDoForm.addEventListener("submit", handleToDoSubmit);
const savedToDos = localStorage.getItem(TODOS_KEY);

if (savedToDos){
    const parsedToDos = JSON.parse(savedToDos); // array 형태
    toDos = parsedToDos; // 새로운 toDos를 입력하면 더이상 빈 값이 아님
    parsedToDos.forEach(paintToDo);

}

// 获取dom元素
var droptarget = document.getElementById("drop")
function handleEvent(event) {
    // 阻止事件的默认行为
    event.preventDefault();
    if (event.type === 'drop') {
      // 文件进入并松开鼠标,文件边框恢复正常
      droptarget.style.borderColor = 'blue'
      for (let file of event.dataTransfer.files) {
        // 把文件保存到文件数组中
        fileArr.push(file)
        // 初始化文件
        filesToBlod(file)
      }
    } else if (event.type === 'dragleave') {
      // 离开时边框恢复
      droptarget.style.borderColor = 'blue'
    } else {
      // 进入边框变为红色
      droptarget.style.borderColor = 'red'
    }
}

droptarget.addEventListener("dragenter", handleEvent);
droptarget.addEventListener("dragover", handleEvent);
droptarget.addEventListener("drop", handleEvent);
droptarget.addEventListener("dragleave", handleEvent);


// Tabbed Menu
function openMenu(evt, menuName) {
  var i, x, tablinks;
  x = document.getElementsByClassName("menu");
  for (i = 0; i < x.length; i++) {
     x[i].style.display = "none";
  }
  tablinks = document.getElementsByClassName("tablink");
  for (i = 0; i < x.length; i++) {
     tablinks[i].className = tablinks[i].className.replace(" w3-red", "");
  }
  document.getElementById(menuName).style.display = "block";
  evt.currentTarget.firstElementChild.className += " w3-red";
}
document.getElementById("myLink").click();



function openLogin() {
  document.getElementById("myForm").style.display = "block";
  document.getElementById("signupForm").style.display = "none";
}
function closeForm() {
  document.getElementById("myForm").style.display = "none";
}

function openSignup() {
  document.getElementById("signupForm").style.display = "block";
  document.getElementById("myForm").style.display = "none";
}
function closeSignup() {
  document.getElementById("signupForm").style.display = "none";
}



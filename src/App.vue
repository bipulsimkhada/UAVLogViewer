<template>
    <div id="app" style="height: 100%;">
        <router-view />
        <ChatComponent />
    </div>
</template>

<script>
import ChatComponent from './components/ChatComponent.vue'

export default {
    name: 'App',
    components: {
        ChatComponent
    }
}
</script>

<style>
@import url('https://fonts.googleapis.com/css?family=Nunito+Sans&display=swap');

#app {
    font-family: 'Nunito Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

html,
body {
    overscroll-behavior-y: contain;
    margin: 0;
    height: 100%;
}
</style>

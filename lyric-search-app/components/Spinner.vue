<template>
  <div class="container" :style="`width: ${size}px; height: ${size}px;`">
    <transition>
      <span v-if="state === 'spinner'" :style="{ '--color': color }">
        <div class="bar1"></div>
        <div class="bar2"></div>
        <div class="bar3"></div>
        <div class="bar4"></div>
        <div class="bar5"></div>
        <div class="bar6"></div>
        <div class="bar7"></div>
        <div class="bar8"></div>
        <div class="bar9"></div>
        <div class="bar10"></div>
        <div class="bar11"></div>
        <div class="bar12"></div>
      </span>
    </transition>
    <transition>
      <div
        class="checkmark"
        v-if="state === 'done'"
        :style="{ '--color': color }"
      ></div>
    </transition>
  </div>
</template>

<script>
export default {
  name: 'Spinner',
  props: {
    color: {
      type: String,
      default: '#555',
    },
    state: {
      type: String,
      default: 'spinner',
    },
    size: {
      type: Number,
      default: 50,
    }
  }
}
</script>

<style scoped>
div.container {
  position: relative;
  display: inline-block;
  padding: 5px;
}

div.container div[class^='bar'] {
  width: 6%;
  height: 16%;
  background: var(--color);
  position: absolute;
  left: 49%;
  top: 43%;
  opacity: 0;
  border-radius: 50px;
  box-shadow: 0 0 3px rgba(0, 0, 0, 0.2);
  animation: fade 1s linear infinite;
}

@keyframes fade {
  from { opacity: 1; }
  to { opacity: 0.25; }
}

div.container div.bar1 {
  transform: rotate(0deg) translate(0, -130%);
  animation-delay: 0s;
}

div.container div.bar2 {
  transform: rotate(30deg) translate(0, -130%);
  animation-delay: -0.9167s;
}

div.container div.bar3 {
  transform: rotate(60deg) translate(0, -130%);
  animation-delay: -0.833s;
}
div.container div.bar4 {
  transform: rotate(90deg) translate(0, -130%);
  animation-delay: -0.7497s;
}
div.container div.bar5 {
  transform: rotate(120deg) translate(0, -130%);
  animation-delay: -0.667s;
}
div.container div.bar6 {
  transform: rotate(150deg) translate(0, -130%);
  animation-delay: -0.5837s;
}
div.container div.bar7 {
  transform: rotate(180deg) translate(0, -130%);
  animation-delay: -0.5s;
}
div.container div.bar8 {
  transform: rotate(210deg) translate(0, -130%);
  animation-delay: -0.4167s;
}
div.container div.bar9 {
  transform: rotate(240deg) translate(0, -130%);
  animation-delay: -0.333s;
}
div.container div.bar10 {
  transform: rotate(270deg) translate(0, -130%);
  animation-delay: -0.2497s;
}
div.container div.bar11 {
  transform: rotate(300deg) translate(0, -130%);
  animation-delay: -0.167s;
}
div.container div.bar12 {
  transform: rotate(330deg) translate(0, -130%);
  animation-delay: -0.0833s;
}
.checkmark {
  display: inline-block;
  position: relative;
  left: -5px;
  transform: rotate(45deg);
  height: 12px;
  width: 6px;
  margin-left: 60%;
  border-bottom: 3px solid var(--color);
  border-right: 3px solid var(--color);
}

.v-enter-active,
.v-leave-active {
  transition: opacity 0.5s ease;
}
.v-enter-from,
.v-leave-to {
  opacity: 0;
}
</style>

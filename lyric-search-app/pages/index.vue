<template>
  <div>
    <header
      :class="[
        'w-full bg-white transition-all ease-in-out',
        hasSearched
          ? 'fixed top-0 left-0 z-10 shadow py-4'
          : 'h-screen flex items-center justify-center'
      ]"
    >
      <div class="w-full max-w-xl px-4"
        :class="hasSearched
          ? 'flex items-center space-x-4'
          : 'flex flex-col items-center justify-center space-y-4 h-full'"
      >
        <!-- Logo -->
        <img
          src="/logo/logo.svg"
          alt="My App Logo"
          class="mx-auto"
          :class="hasSearched
            ? 'h-10 cursor-pointer'
            : 'mb-20 h-40'"
          onclick="location.reload()"
        />

        <!-- Search Input -->
        <input
          v-model="query"
          autofocus
          @keyup.enter="onSearch"
          :placeholder="hasSearched
            ? 'Search again…'
            : 'Search lyrics, song, artist…'"
          class="w-full border border-gray-300 rounded-lg px-4 py-2
                 focus:outline-none focus:ring focus:border-blue-300"
        />
      </div>
    </header>

    <!-- Spacer for sticky header -->
    <div v-if="hasSearched" class="h-24"></div>

    <!-- Results List -->
    <main class="max-w-xl mx-auto space-y-4 px-4">
      <div
        v-for="track in tracks"
        :key="track.id"
        @click="selectTrack(track)"
        class="bg-white p-4 rounded-lg shadow cursor-pointer hover:shadow-lg transition"
      >
        <h2 class="text-lg font-semibold">
          {{ track.trackName }} – {{ track.artistName }}
        </h2>
        <p class="text-sm text-gray-500 mb-2">{{ track.albumName }}</p>
        <p class="text-sm line-clamp-3">{{ track.plainLyrics }}</p>
      </div>

      <!-- Loading Spinner -->
      <div v-if="isLoading" class="flex justify-center py-6">
        <svg
          class="animate-spin h-8 w-8 text-gray-600"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25"/>
          <path fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" class="opacity-75"/>
        </svg>
      </div>
    </main>

    <!-- Lyrics Modal -->
    <div
      v-if="selectedTrack"
      class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-20"
      @click.self="closeModal"
    >
      <div class="bg-white rounded-lg max-w-lg w-full max-h-[80vh] overflow-hidden flex flex-col">
        <!-- Header -->
        <div class="flex justify-between items-center border-b px-4 py-2">
          <h3 class="font-semibold">{{ selectedTrack.trackName }} – {{ selectedTrack.artistName }}</h3>
          <button @click="closeModal" class="text-gray-500 hover:text-gray-800 text-xl leading-none">
            &times;
          </button>
        </div>

        <!-- Tab Nav -->
        <div class="flex border-b">
          <button
            @click="tab = 'plain'"
            :class="tab === 'plain' ? activeTabClass : inactiveTabClass"
            class="flex-1 text-center py-2"
          >
            Plain Lyrics
          </button>
          <button
            @click="tab = 'synced'"
            :class="tab === 'synced' ? activeTabClass : inactiveTabClass"
            class="flex-1 text-center py-2"
          >
            Synced Lyrics
          </button>
        </div>

        <!-- Content -->
        <div class="p-4 overflow-y-auto flex-1 text-sm whitespace-pre-wrap">
          <div v-if="tab === 'plain'">
            {{ selectedTrack.plainLyrics }}
          </div>
          <div v-else>
            {{ selectedTrack.syncedLyrics }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { Client } from '@gradio/client'

const query        = ref('')
const rawResults   = ref([])
const tracks       = ref([])
const page         = ref(0)
const isLoading    = ref(false)
const hasSearched  = ref(false)

const selectedTrack = ref(null)
const tab           = ref('plain')

// Tailwind classes for tabs
const activeTabClass   = 'border-b-2 border-blue-500 font-semibold'
const inactiveTabClass = 'text-gray-600'

// Gradio client handle
let grClient = null

// initialize client & scroll listener
onMounted(async () => {
  grClient = await Client.connect('ernestchu/lyric-search')
  window.addEventListener('scroll', onScroll)
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', onScroll)
})

// user hits Enter
async function onSearch() {
  if (!query.value.trim()) return

  hasSearched.value = true
  page.value        = 0
  tracks.value      = []
  isLoading.value   = true

  try {
    const { data } = await grClient.predict('/search', { query: query.value })
    rawResults.value = JSON.parse(data[0].value)
    await fetchResults()
  } catch (err) {
    console.error(err)
  } finally {
    isLoading.value = false
  }
}

// paginate through rawResults
async function fetchResults() {
  const start = page.value * 10
  if (start >= rawResults.value.length) return

  isLoading.value = true
  try {
    const slice = rawResults.value.slice(start, start + 10)
    const metas = await Promise.all(
      slice.map(async ([id, score, lcs]) => {
        const res  = await fetch(`https://lrclib.net/api/get/${id}`)
        const json = await res.json()
        return { ...json, score, lcs }
      })
    )
    tracks.value.push(...metas)
    page.value++
  } catch (err) {
    console.error(err)
  } finally {
    isLoading.value = false
  }
}

// infinite scroll
function onScroll() {
  if (isLoading.value) return
  const nearBottom = window.innerHeight + window.scrollY >= document.body.offsetHeight - 100
  if (nearBottom) fetchResults()
}

// open modal
function selectTrack(track) {
  selectedTrack.value = track
  tab.value           = 'plain'
}

// close modal
function closeModal() {
  selectedTrack.value = null
}
</script>

<style>
.line-clamp-3 {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>

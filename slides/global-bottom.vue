<script setup>
import { computed } from 'vue'
import { useNav } from '@slidev/client'

// Persistent section breadcrumb (option C). Each section's opening slide is
// tagged with a `section:` frontmatter; every following slide walks back to the
// nearest tag, so the current section stays visible — even when jumping via the
// skip-to-recap links.
const { slides, currentPage, currentLayout } = useNav()

const section = computed(() => {
  const all = slides?.value || []
  const idx = (currentPage?.value || 1) - 1
  for (let i = Math.min(idx, all.length - 1); i >= 0; i--) {
    const s = all[i]?.meta?.slide?.frontmatter?.section
    if (s) return s
  }
  return ''
})
</script>

<template>
  <footer
    v-if="section && currentLayout !== 'section' && currentLayout !== 'cover'"
    class="section-crumb"
  >
    {{ section }}
  </footer>
</template>

<style scoped>
.section-crumb {
  position: absolute;
  bottom: 0.6rem;
  left: 0.8rem;
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.4;
}
</style>

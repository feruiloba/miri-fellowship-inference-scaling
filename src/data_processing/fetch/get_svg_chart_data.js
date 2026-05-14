(id) => {
    const el = document.getElementById(id);
    if (!el) return { axis_labels: [], value_labels: [], bar_rects: [] };

    const axis_labels = [];
    const value_labels = [];
    el.querySelectorAll('svg text').forEach(e => {
        const t = e.textContent.trim();
        if (!t) return;
        const xAttr = e.getAttribute('x');
        if (xAttr === null) {
            axis_labels.push(t);
        } else {
            value_labels.push({
                text: t,
                x: parseFloat(xAttr),
                y: parseFloat(e.getAttribute('y') || '0'),
            });
        }
    });

    const bar_rects = [];
    el.querySelectorAll('rect[data-testid]').forEach(r => {
        const testid = r.getAttribute('data-testid');
        const height = parseFloat(r.getAttribute('height') || '0');
        // Walk up to find cumulative y transform from parent <g> elements
        let translateY = 0;
        let parent = r.parentElement;
        while (parent && parent !== el) {
            const tf = parent.getAttribute('transform');
            if (tf) {
                const m = tf.match(/translate\([\s\d.]+,\s*([\d.]+)\)/);
                if (m) translateY += parseFloat(m[1]);
            }
            parent = parent.parentElement;
        }
        bar_rects.push({ testid, height, translateY });
    });

    return { axis_labels, value_labels, bar_rects };
}

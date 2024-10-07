
export default function Button({type, name, text, func}) {
  return (
    <div>
      <button type={type} className={text} onClick={func}>{name}</button>
    </div>
  )
}
